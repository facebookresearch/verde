import ast
import os
import time
import pickle
from collections import OrderedDict
from logging import getLogger

import numpy as np
import torch
from scipy import stats

from src.train.model import TransformerModel
from src.utils import to_cuda

logger = getLogger()


class SecretCheck(object):
    def __init__(self, trainer, dataset):
        self.trainer = trainer
        self.params = trainer.params
        self.orig_A, self.orig_b = dataset.orig_A, dataset.orig_b
        self.secret_recovery = { 'success': [] }

    def match_secret(self, guess, method_name):
        '''
        Takes an int or bool (binary) list or array as secret guess and check against the original tiny dataset. 
        '''
        guess = np.array(guess).astype(int)
        if self.params.secret_type in ['gaussian', 'binomial']:
            # only check if nonzeros are identified for gaussian and binomial secrets
            matched = np.all((self.params.secret != 0) == (guess != 0))
        elif self.orig_A is None: # Old data, original dataset not available. Directly check the secret. 
            matched = np.all(self.params.secret == guess)
        else:
            err_pred = (self.orig_A @ guess - self.orig_b) % self.params.Q
            err_pred[err_pred > self.params.Q // 2] -= self.params.Q
            matched = np.std(err_pred) < 2*self.params.sigma
        if matched: 
            logger.info(f'{method_name}: all bits in secret have been recovered!')
            if method_name not in self.secret_recovery['success']:
                self.secret_recovery['success'].append(method_name)
                self.trainer.secret_match = True
            return True

    def match_secret_iter(self, idx_list, sorted_idx_with_scores, method_name):
        ''' 
        Takes a list of indices sorted by scores (descending, high score means more likely to be 1)
        and iteratively matches the secret. 
        '''
        self.secret_recovery[method_name] = sorted_idx_with_scores or idx_list
        guess = np.zeros(self.params.N)
        for i in range(min(self.params.N // 5, len(idx_list))): # sparse assumption
            guess[idx_list[i]] = 1
            if self.match_secret(guess, method_name):
                return True
        logger.info(f'{method_name}: secret not predicted.')
        return False
    
    def add_log(self, k, v):
        self.secret_recovery[k] = v

    def store_results(self, path, epoch):
        try:
            pickle.dump(self.secret_recovery, open(os.path.join(path, f'secret_recovery_{epoch}.pkl'), 'wb'))
        except Exception as e:
            logger.info(f'secret recovery: {self.secret_recovery}')
            logger.info(f'Exception when saving secret_recovery details: {e}')


class Evaluator(object):
    def __init__(self, trainer, test_dataloader):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.iterator = test_dataloader
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        self.secret_check = SecretCheck(trainer, test_dataloader.dataset)

    def run_all_evals(self):
        """
        Run all evaluations.
        """
        scores = OrderedDict({"epoch": self.trainer.epoch})

        with torch.no_grad():
            encoder = (
                self.modules["encoder"].module
                if self.params.multi_gpu
                else self.modules["encoder"]
            )
            decoder = (
                self.modules["decoder"].module
                if self.params.multi_gpu and hasattr(self.modules["decoder"], 'module')
                else self.modules["decoder"]
            )
            encoder.eval()
            decoder.eval()
            self.run_distinguisher(encoder, decoder)
            self.run_direct_recovery(encoder, decoder)
            self.recover_secret_from_crossattention(encoder, decoder, scores) # cross attention (+ circular regression)
            self.hybrid()
        self.secret_check.store_results(self.params.dump_path, self.trainer.epoch)
        return scores

    def ordered_idx_from_scores(self, secret_scores):
        ''' Takes bit-wise scores (length N) and return sorted list<(idx, score)> and sorted list<idx>. '''
        idx_with_scores = list(enumerate(secret_scores)) # a list of (idx, score)
        sorted_idx_by_scores = sorted(idx_with_scores, key=lambda item: item[1], reverse=True) # descending
        return sorted_idx_by_scores, [t[0] for t in sorted_idx_by_scores]

    def hybrid(self):
        '''
        Hybrid secret recovery that combines direct secret recovery, distinguisher and CA
        '''
        methods_dict = {
            'direct': self.direct_results, 
            'distinguisher': self.distinguisher_results, 
            'ca': self.ca_results, 
        }
        combos = [['direct', 'ca'], ['direct', 'distinguisher'], ['ca', 'distinguisher'], ['direct', 'ca', 'distinguisher']]
        for combo in combos:
            logger.info(f'Hybrid: {", ".join(combo)}')
            self.hybrid_sub([methods_dict[m] for m in combo], ", ".join(combo))

    def hybrid_sub(self, methods, combo_name):
        for results in methods:
            if max(results) == 0: # the scores are non-negative. Hybrid on this combo is useless. 
                return None

        sum_and_max = np.zeros((4,self.params.N))
        for results in methods:
            # Normalized, sum and max
            sum_and_max[0] += results/max(results)
            sum_and_max[1] = np.max((sum_and_max[1], results/max(results)), axis=0)
            # Ranking, sum and max
            rank = stats.rankdata(results, method='min')
            sum_and_max[2] += rank
            sum_and_max[3] = np.max((sum_and_max[3], rank), axis=0)

        for i, name in enumerate(['Sum Normalized', 'Max Normalized', 'Sum Rank', 'Max Rank']):
            idx_w_scores, indices = self.ordered_idx_from_scores(sum_and_max[i])
            self.secret_check.match_secret_iter(indices, idx_w_scores, f'{combo_name} - {name}')


    ########################################################
    # CODE TO RUN DIRECT SECRET RECOVERY AND DISTINGUISHER #
    ########################################################
    def run_beam_generation(self, x1_, len1_, encoder, decoder):
        # Run beam generation to get output. 
        encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
        _, _, generations= decoder.generate_beam(encoded.transpose(0, 1), len1_,
                                                    beam_size=self.params.beam_size,
                                                    length_penalty=self.params.beam_length_penalty,
                                                    early_stopping=self.params.beam_early_stopping,
                                                    max_len=self.params.max_output_len)
        beam_log = []
        for i in range(len(generations)):
            sorted_hyp = sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)
            if len(sorted_hyp) == 0:
                beam_log.append(0)
            else:
                _, hyp = sorted_hyp[0]
                output = [self.trainer.env.id2word[wid] for wid in hyp[1:].tolist()]
                try:
                    beam_log.append(self.env.output_encoder.decode(output)[0])
                except Exception as e:
                    beam_log.append(-1)
        return beam_log

    def predict_outputs(self, A, encoder, decoder, intermediate=False):
        ''' 
        if intermediate is False then output integers
        if intermediate is True then output distributions 
        '''
        preds = []
        # Encodes data in format expected by model
        encA = self.env.input_encoder.encode(A)
        encA = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in encA]
        for k in range(0, len(encA), self.params.batch_size):
            x = encA[k:k+self.params.batch_size]
            x1, len1 = self.env.batch_sequences(x)
            x1_, len1_ = to_cuda(x1, len1)
            preds.extend(self.run_beam_generation(x1_, len1_, encoder, decoder))
        return np.array(preds)

    def run_direct_recovery(self, encoder, decoder):
        self.direct_results = np.zeros(self.params.N)
        invert = np.vectorize(lambda x: 1 - x)
        logger.info('Starting Direct Method')

        for K in np.random.randint(self.params.Q//4, 3*self.params.Q//4, 15):
            logger.info(f'Direct:  K={K}')
            specialA = np.identity(self.params.N, dtype=np.int64) * K
            pred_final = self.predict_outputs(specialA, encoder, decoder)
            try:
                pred_softmax = torch.nn.Softmax(dim=0)(torch.Tensor(pred_final)).detach().cpu().numpy()
            except:
                logger.info('Error in softmax prediction, secret decoding failed.')
                continue

            # 3 methods of testing for matching: mean, mode, and softmax mean
            pred_bin1 = np.vectorize(lambda x: 0 if x > np.mean(pred_final) else 1)(pred_final) 
            pred_bin2 = np.vectorize(lambda x: 0 if x != stats.mode(pred_final)[0][0] else 1)(pred_final)
            pred_bin3 = np.vectorize(lambda x: 0 if x > np.mean(pred_softmax) else 1)(pred_softmax)

            # Match list
            for match_vec in [pred_bin1, pred_bin2, pred_bin3]:
                self.secret_check.match_secret(match_vec, 'Direct')
                self.secret_check.match_secret(invert(match_vec), 'Direct')
            self.direct_results += pred_softmax

        idx_w_scores, indices = self.ordered_idx_from_scores(self.direct_results)
        self.secret_check.match_secret_iter(indices, idx_w_scores, 'Direct')
        
    def run_distinguisher(self, encoder, decoder):
        self.distinguisher_results = np.zeros(self.params.N)
        logger.info(f'Starting Distinguisher Method')
        num_samples = self.params.distinguisher_size
        # Get the A (bkz reduced) and run through the model. 
        A_s = np.array(self.iterator.dataset.getbatchA(num_samples))
        lwe_preds0 = self.predict_outputs(A_s, encoder, decoder, intermediate=True)
        # Prepare the random values to add to each coordinate of A. 
        # The first half in (0.3q, 0.4q), the second half in (0.6q, 0.7q)
        add_rand = np.random.randint(3*self.params.Q//10, 2*self.params.Q//5, size=num_samples//2)
        add_rand = np.concatenate([add_rand, add_rand*-1])

        lwe_preds = []
        for i in range(self.params.N):
            # Get the A' and run through the model. 
            A_s[:,i] = (A_s[:,i] + add_rand) % self.params.Q # add a random value to the ith coordinate of A
            lwe_preds.append(self.predict_outputs(A_s, encoder, decoder, intermediate=True))
            A_s[:,i] = (A_s[:,i] - add_rand) % self.params.Q # revert change

        # Recover secret. Higher earth mover's distance -> bit nonzero. Higher mean abs diff -> bit nonzero. 
        self.secret_check.add_log('distinguisher_orig', lwe_preds0)
        self.secret_check.add_log('distinguisher_bits', lwe_preds)
        emd_func = 'emd', stats.wasserstein_distance
        mean_func = 'mean', lambda x,y: np.mean(abs(x-y))
        for func_name, get_diff in [emd_func, mean_func]:
            logger.info(f"Distinguishing 0s using the {func_name}. ")
            for i in range(self.params.N):
                self.distinguisher_results[i] = get_diff(lwe_preds[i], lwe_preds0)
            if self.params.secret_type == 'ternary':
                try:
                    self.secret_check.add_log(f'Distinguisher Method {func_name}', self.distinguisher_results)
                    ternary_dist = TernaryDistinguisher(self.secret_check, func_name)
                    ternary_dist.run(lwe_preds, self.distinguisher_results, emd_func)
                    ternary_dist.run(lwe_preds, self.distinguisher_results, mean_func)
                except Exception as e:
                    logger.info(f'Exception in ternary secret distinguisher: {e}')
            else:
                sorted_idx_with_scores, indices = self.ordered_idx_from_scores(self.distinguisher_results)
                self.secret_check.match_secret_iter(indices, sorted_idx_with_scores, f'Distinguisher Method {func_name}')

    ############################################
    # CODE TO RUN CROSS ATTENTION AND CIRC REG #
    ############################################
    def recover_secret_from_crossattention(self, encoder, decoder, scores):
        """
        Guess the secret from the cross attention matrix
        """
        TransformerModel.STORE_OUTPUTS = True

        # iterator
        ca_results = np.zeros(self.params.N), np.zeros(self.params.N)
        xe_loss = []
        for (x1, len1), (x2, len2), nb_ops in self.iterator:
            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                alen[:, None] < len2[None] - 1
            )  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()
            # cuda
            x1_, len1_, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

            # forward
            encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1_,
            )
            word_scores, loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
            )
            xe_loss.append(loss.item())

            for layerId in [0, 1]:
                ca_scores = torch.stack(decoder.layers[layerId].cross_attention.outputs)
                assert ca_scores.ndim == 5
                guessed_idxs = self.guess_from_scores(ca_scores, 3)
                for j in guessed_idxs:
                    ca_results[layerId][j] += 1
        
        self.ca_results = ca_results[0] + ca_results[1]
        for layerId in [0, 1]:
            sorted_idx_by_count, indices = self.ordered_idx_from_scores(ca_results[layerId])
            self.secret_check.match_secret_iter(indices, sorted_idx_by_count, 'CA')

        scores["valid_xe_loss"] = np.mean(xe_loss)
        self.secret_check.add_log('xe_loss', np.mean(xe_loss))
        TransformerModel.STORE_OUTPUTS = False


    def guess_from_scores(self, ca_scores, threshold_coeff):
        ''' Propose secret indices that are likely nonzero '''
        loops, bs, heads, ylen, xlen = ca_scores.shape
        guessed = []
        for h in range(heads):
            for y in range(ylen):
                mean_scores = ca_scores[:,:,h,y].view(-1, xlen).mean(0)
                threshold = threshold_coeff*mean_scores.mean()
                stems = torch.where(mean_scores > threshold)[0].numpy()
                candidates = (stems[(stems!=0)&(stems!=xlen-1)]-1)//self.env.input_encoder.int_len
                guessed.extend(np.unique(candidates))
                
        return guessed


class TernaryDistinguisher(object):
    def __init__(self, secret_check, func_name):
        self.secret_check = secret_check
        self.params = secret_check.params
        self.func0 = func_name
    
    def check_ternary(self, nonzeros, clique0, clique1):
        """
        nonzeros: an array of secret indices
        clique0, clique1: partition of these indices. Their union should be integers in [0, len(nonzeros))
        """
        guess = np.zeros(self.params.N)
        guess[nonzeros[list(clique0)]] = 1
        guess[nonzeros[list(clique1)]] = -1
        matching = self.secret_check.match_secret(guess, 'Distinguisher Method')
        return matching or self.secret_check.match_secret(guess*-1, 'Distinguisher Method')

    def run(self, lwe_preds, diffs, func):
        """
        Runs the ternary secret distinguisher: guesses nonzero bits and splits them into two sets for opposite signs. 
        This is achieved in 2 steps:
        1. Calculate the distance of distinguisher predictions (two halves) for each pair of indices. 
           Recall that the first half added (0.3q, 0.4q) to the input, the second half substracted the same values.
           The bits with the same sign should have lower values on the first and second half, respectively. 
        2. Glide through possible hamming weights and form bipartitions according to the distances.
        """
        func_name, diff_func = func
        self.func1 = func_name
        logger.info(f"Distinguishing +/-1s using the {func_name}. ")
        idx_sorted = np.argsort(diffs) # sorted secret indices. Closer to the end means more likely nonzero. 
        # cheat, just for debugging and logging
        logger.info(f'Real secret: -1: {set(np.where(self.params.secret == -1)[0])}, 1: {set(np.where(self.params.secret == 1)[0])}')
        if set(idx_sorted[-self.params.hamming:]) != set(np.where(self.params.secret != 0)[0]):
            logger.info(f'Nonzero bits not identified. Guess: {idx_sorted[-self.params.hamming:]}')
            return False

        # eliminate the cases of h=1 and 2
        if self.check_ternary(idx_sorted[-1:], [0], []):
            return True
        if self.check_ternary(idx_sorted[-2:], [0,1], []) or self.check_ternary(idx_sorted[-2:], [0], [1]):
            return True
        
        dists = np.zeros((self.params.N, self.params.N))
        half = self.params.distinguisher_size // 2
        for i in range(self.params.N):
            for j in range(i):
                dists[i, j] = diff_func(lwe_preds[i][half:], lwe_preds[j][half:])
                dists[j, i] = diff_func(lwe_preds[i][:half], lwe_preds[j][:half])

        # for each hamming weight, make secret guesses by forming bipartitions of the nonzero bits.
        for h in range(3, self.params.N // 5): # sparse assumption
            nonzeros = idx_sorted[-h:]
            dist_mats = [np.zeros((h, h)), np.zeros((h, h))]
            for i in range(h):
                for j in range(i):
                    idx1, idx2 = nonzeros[i], nonzeros[j]
                    dist_mats[0][i,j], dist_mats[1][i,j] = dists[idx1,idx2], dists[idx2,idx1]
            dist_mats.append(dist_mats[0]+dist_mats[1])
            for dist_mat in dist_mats:
                if self.bipartition(dist_mat, nonzeros):
                    return True
        
    def bipartition(self, dist_mat, nonzeros):
        """
        Given a distance matrix and an array of nonzero bits, 
        form bipartitions of the nonzero bits s.t. the distance within the partitions are low,
        and the distance across the partitions are high. 
        """
        def fm(clique0, clique1):
            """
            This algorithm is inspired by the Fiduccia-Mattheyses Partitioning Algorithm.
            Relaxed the condition to allow more randomness. 
            """
            # check the correctness of the initial partition
            ternary_log = [[nonzeros[list(clique0)], nonzeros[list(clique1)]]]
            if self.check_ternary(nonzeros, clique0, clique1):
                self.secret_check.add_log(f'ternary_{self.func0}_{self.func1}', ternary_log)
                return True
            # initialize the metric: the average dist across cliques over the average dist within cliques
            changed = True
            in_sum, cross_sum, in_num, cross_num = 0,0,0,0
            for ii in range(len(nonzeros)):
                for jj in range(ii):
                    if (ii in clique0 and jj in clique0) or (ii in clique1 and jj in clique1):
                        in_sum += dist_mat[ii][jj]
                        in_num += 1
                    else:
                        cross_sum += dist_mat[ii][jj]
                        cross_num += 1
            if cross_sum == 0 or in_sum == 0:
                return False
            before_move = cross_sum * in_num / cross_num / in_sum
            # try swapping for a finite number of times
            for _ in range(10):
                for ii in range(len(nonzeros)):
                    clique0scores = [dist_mat[max(ii,jj)][min(ii,jj)] for jj in clique0 if ii!=jj]
                    clique1scores = [dist_mat[max(ii,jj)][min(ii,jj)] for jj in clique1 if ii!=jj]
                    # keep the move if it increases 
                    move = sum(clique0scores)-sum(clique1scores), len(clique0scores)-len(clique1scores)
                    if ii in clique0 and len(clique0) != 1: # move from clique0 to clique1
                        after_move = (cross_sum+move[0]) * (in_num-move[1]) / (cross_num+move[1]) / (in_sum-move[0])
                        clique0.remove(ii)
                        clique1.add(ii)
                        if self.check_ternary(nonzeros, clique0, clique1):
                            ternary_log.append([nonzeros[list(clique0)], nonzeros[list(clique1)]])
                            self.secret_check.add_log(f'ternary_{self.func0}_{self.func1}', ternary_log)
                            return True
                        if after_move < before_move: # revert
                            clique1.remove(ii)
                            clique0.add(ii)
                        else:
                            ternary_log.append([nonzeros[list(clique0)], nonzeros[list(clique1)]])
                    elif ii in clique1 and len(clique1) != 1: # move from clique1 to clique0
                        after_move = (cross_sum-move[0]) * (in_num+move[1]) / (cross_num-move[1]) / (in_sum+move[0])
                        clique1.remove(ii)
                        clique0.add(ii)
                        if self.check_ternary(nonzeros, clique0, clique1):
                            ternary_log.append([nonzeros[list(clique0)], nonzeros[list(clique1)]])
                            self.secret_check.add_log(f'ternary_{self.func0}_{self.func1}', ternary_log)
                            return True
                        if after_move < before_move:
                            clique0.remove(ii)
                            clique1.add(ii)
                        else:
                            ternary_log.append([nonzeros[list(clique0)], nonzeros[list(clique1)]])
            if len(nonzeros) == self.params.hamming:
                self.secret_check.add_log(f'ternary_{self.func0}_{self.func1}', ternary_log)

        # initialize the partition using a greedy algorithm, using distance as the heuristic
        x1, y1 = np.unravel_index(np.argsort(dist_mat, axis=None), dist_mat.shape)
        cliques = [] # a list of disjoint sets
        for i,j in zip(x1,y1):
            if j >= i:
                continue
            if len(cliques) == 0: 
                cliques.append(set([i, j]))
            else:
                membership = [None, None]
                for clique_id in range(len(cliques)):
                    if i in cliques[clique_id]:
                        membership[0] = clique_id
                    if j in cliques[clique_id]:
                        membership[1] = clique_id
                # if neither elements found, create a new clique
                if membership[0] is None and membership[1] is None:
                    cliques.append(set([i, j]))
                # if one element belongs to a clique, add the other element to that clique
                elif membership[0] is not None and membership[1] is None:
                    cliques[membership[0]].add(j)
                elif membership[0] is None and membership[1] is not None:
                    cliques[membership[1]].add(i)
                # if both elements are found in cliques, merge the cliques
                elif membership[0] != membership[1]:
                    cliques[membership[0]] = cliques[membership[0]].union(cliques[membership[1]])
                    del cliques[membership[1]]
                    
            # We have finished processing the pair. Check if we are done.
            if len(cliques) == 1 and np.sum([len(clique) for clique in cliques]) == len(nonzeros)-1:
                # the algorithm is suggesting that all indices have the same sign, or only one index has a different sign
                if self.check_ternary(nonzeros, range(len(nonzeros)), []):
                    return True
                return fm(cliques[0], set(range(len(nonzeros)))-cliques[0])
            
            if len(cliques) == 2 and np.sum([len(clique) for clique in cliques]) == len(nonzeros):
                return fm(cliques[0], cliques[1])


        