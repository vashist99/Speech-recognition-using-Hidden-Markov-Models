import numpy as np
from scipy.stats import multivariate_normal
import os
import argparse
# from htk import HTKFile
import librosa
from pydub import AudioSegment

import subprocess
import struct

class HTKFile:
    """ Class to load binary HTK file.

        Details on the format can be found online in HTK Book chapter 5.7.1.

        Not everything is implemented 100%, but most features should be supported.

        Not implemented:
            CRC checking - files can have CRC, but it won't be checked for correctness

            VQ - Vector features are not implemented.
    """

    data = None
    nSamples = 0
    nFeatures = 0
    sampPeriod = 0
    basicKind = None
    qualifiers = None
    endian = '>'

    def load(self, filename):
        """ Loads HTK file.

            After loading the file you can check the following members:

                data (matrix) - data contained in the file

                nSamples (int) - number of frames in the file

                nFeatures (int) - number if features per frame

                sampPeriod (int) - sample period in 100ns units (e.g. fs=16 kHz -> 625)

                basicKind (string) - basic feature kind saved in the file

                qualifiers (string) - feature options present in the file

        """
        with open(filename, "rb") as f:

            header = f.read(12)
            self.nSamples, self.sampPeriod, sampSize, paramKind = struct.unpack(">iihh", header)
            if self.nSamples<0 or self.sampPeriod<0 or sampSize<0:
                self.endian = '<'
                self.nSamples, self.sampPeriod, sampSize, paramKind = struct.unpack(self.endian+"iihh", header)
            basicParameter = paramKind & 0x3F

            if basicParameter is 0:
                self.basicKind = "WAVEFORM"
            elif basicParameter is 1:
                self.basicKind = "LPC"
            elif basicParameter is 2:
                self.basicKind = "LPREFC"
            elif basicParameter is 3:
                self.basicKind = "LPCEPSTRA"
            elif basicParameter is 4:
                self.basicKind = "LPDELCEP"
            elif basicParameter is 5:
                self.basicKind = "IREFC"
            elif basicParameter is 6:
                self.basicKind = "MFCC"
            elif basicParameter is 7:
                self.basicKind = "FBANK"
            elif basicParameter is 8:
                self.basicKind = "MELSPEC"
            elif basicParameter is 9:
                self.basicKind = "USER"
            elif basicParameter is 10:
                self.basicKind = "DISCRETE"
            elif basicParameter is 11:
                self.basicKind = "PLP"
            else:
                self.basicKind = "ERROR"

            self.qualifiers = []
            if (paramKind & 0o100) != 0:
                self.qualifiers.append("E")
            if (paramKind & 0o200) != 0:
                self.qualifiers.append("N")
            if (paramKind & 0o400) != 0:
                self.qualifiers.append("D")
            if (paramKind & 0o1000) != 0:
                self.qualifiers.append("A")
            if (paramKind & 0o2000) != 0:
                self.qualifiers.append("C")
            if (paramKind & 0o4000) != 0:
                self.qualifiers.append("Z")
            if (paramKind & 0o10000) != 0:
                self.qualifiers.append("K")
            if (paramKind & 0o20000) != 0:
                self.qualifiers.append("0")
            if (paramKind & 0o40000) != 0:
                self.qualifiers.append("V")
            if (paramKind & 0o100000) != 0:
                self.qualifiers.append("T")

            if "C" in self.qualifiers or "V" in self.qualifiers or self.basicKind is "IREFC" or self.basicKind is "WAVEFORM":
                self.nFeatures = sampSize // 2
            else:
                self.nFeatures = sampSize // 4

            if "C" in self.qualifiers:
                self.nSamples -= 4

            if "V" in self.qualifiers:
                raise NotImplementedError("VQ is not implemented")

            self.data = []
            if self.basicKind is "IREFC" or self.basicKind is "WAVEFORM":
                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        val = struct.unpack_from(self.endian+"h", s, v * 2)[0] / 32767.0
                        frame.append(val)
                    self.data.append(frame)
            elif "C" in self.qualifiers:

                A = []
                s = f.read(self.nFeatures * 4)
                for x in range(self.nFeatures):
                    A.append(struct.unpack_from(self.endian+"f", s, x * 4)[0])
                B = []
                s = f.read(self.nFeatures * 4)
                for x in range(self.nFeatures):
                    B.append(struct.unpack_from(self.endian+"f", s, x * 4)[0])

                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        frame.append((struct.unpack_from(self.endian+"h", s, v * 2)[0] + B[v]) / A[v])
                    self.data.append(frame)
            else:
                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        val = struct.unpack_from(self.endian+"f", s, v * 4)
                        frame.append(val[0])
                    self.data.append(frame)

            if "K" in self.qualifiers:
                print("CRC checking not implememnted...")


class Network:
    def __init__(self):
        self.num_states = 0
        self.transitions = []
        self.nodes = []
        self.initial_probs = []
        self.word_map = {}  # Add this line
        self.transition_probs = {}

    def add_node(self, hmm_name, hmm):
        """Add an HMM node to the network."""
        num_states = len(hmm['states'])
        self.num_states += num_states
        self.nodes.append((hmm_name, hmm))
        state_index = self.num_states
        self.word_map[state_index] = hmm_name
        
        # Initialize initial probabilities for new states
        if not self.initial_probs:
            self.initial_probs = [0.0] * self.num_states
            self.initial_probs[0] = 1.0  # Assume first state is initial
        else:
            self.initial_probs.extend([0.0] * num_states)

    def initial_prob(self, state):
        """Return the initial probability for a given state."""
        if state < len(self.initial_probs):
            return self.initial_probs[state]
        return 0.0
    
    def state_to_word(self, state_index):
        # Find the word corresponding to the given state index
        for index, word in self.word_map.items():
            if state_index >= index and state_index < index + len(self.nodes[index][1]['states']):
                return word
        return None  # Return None if no matching word is found
    
    def prev_states(self, s):
        """
        Returns a list of previous states that can transition to the given state s.
        """
        prev_states = []
        for i, node in enumerate(self.nodes):
            for transition in node[1]['transitions']:
                if transition[1] == s:  # Assuming transition[1] is the 'to' state
                    prev_states.append(i)
        return prev_states
    
    def transition_prob(self, prev_s, s):
        """
        Return the transition probability from state prev_s to state s.
        """
        if (prev_s, s) in self.transition_probs:
            return self.transition_probs[(prev_s, s)]
        else:
            # If no specific transition probability is defined, return a small non-zero value
            return 1e-10  # You may need to adjust this default value

    def add_transition(self, prev_s, s, prob):
        """
        Add a transition probability from state prev_s to state s.
        """
        self.transition_probs[(prev_s, s)] = prob



# def create_network_from_words(words, hmm_set, dictionary):
#     network = Network()

    

#     for word in words:
#         if word not in dictionary:
#             raise ValueError(f"Word '{word}' not found in dictionary.")

#         phoneme_sequence = dictionary[word]

#         for phoneme in phoneme_sequence:
#             # key_list = hmm_set.hmms.keys()
#             for pho in phoneme:
#                 if pho not in hmm_set.hmms:
#                     raise ValueError(f"Phoneme '{phoneme}' not found in HMM set.")

#             # Add HMM for this phoneme to the network
#                 hmm = hmm_set.hmms[pho]
#                 network.add_node(phoneme, hmm)
#                 network.add_transition()

#     return network

def create_network_from_words(words, hmm_set, dictionary):
    network = Network()
    
    for word in words:
        phoneme_sequence = dictionary[word]
        phoneme_sequence = phoneme_sequence[0]
        
        for i, phoneme in enumerate(phoneme_sequence):
            hmm = hmm_set.get_hmm(phoneme)
            network.add_node(phoneme, hmm)
            # print("heheheheheh ",hmm)
            
            if i > 0:
                prev_phoneme = phoneme_sequence[i-1]
                prev_hmm = hmm_set.get_hmm(prev_phoneme)
                
                # Get transition probability from the last state of prev_hmm
                # to the first state of current hmm
                trans_prob = prev_hmm['transitions'][-1][0]
                
                network.add_transition(prev_phoneme, phoneme, trans_prob)
            
            # Add transitions between states within the HMM
            for j in range(len(hmm['states']) - 1):
                from_state = f"{phoneme}.{j}"
                to_state = f"{phoneme}.{j+1}"
                trans_prob = hmm['transitions'][j][j+1]
                network.add_transition(from_state, to_state, trans_prob)
    
    return network






class HMMSet:
    def __init__(self):
        self.hmms = {}
        self.transition_matrices = {}
        self.emission_matrices = {}
        self.pkind = None  # Parameter kind (e.g., MFCC)
        self.swidth = []  # Stream widths
        self.numLogHMM = 0  # Number of logical HMMs
        self.numPhyHMM = 0  # Number of physical HMMs
        self.variance_macros = {} 

    def add_hmm(self, name, states, transitions, emissions):
        self.hmms[name] = {
            'states': states,
            'transitions': transitions,
            'emissions': emissions
        }
        self.transition_matrices[name] = np.array(transitions)
        self.emission_matrices[name] = np.array(emissions)
        self.numPhyHMM += 1
        self.numLogHMM += 1

    #one time def
    def get_hmm(self, name):
        return self.hmms.get(name)

    #one time def
    def set_parameter_kind(self, pkind):
        self.pkind = pkind

    #one time def
    def set_stream_widths(self, widths):
        self.swidth = widths

    # def emission_prob(self, state, observation):
    #     if isinstance(state, int):
    #         # Handle state as an integer index
    #         for hmm_name, hmm in self.hmms.items():
    #             print("state: ",state,len(hmm['states']))
    #             if state < len(hmm['states']):
    #                 state_info = hmm['states'][state]
    #                 break
    #             else:
    #                 # raise ValueError(f"State index {state} not found in any HMM")
    #                 continue
                
    #     else:
    #         # Handle state as a string (original implementation)
    #         hmm_name = state.split('.')[0]
    #         state_num = int(state.split('.')[1])
    #         hmm = self.hmms[hmm_name]
    #         state_info = hmm['states'][state_num]
        
    #     total_prob = 0.0
        
    #     for mixture in state_info['mixtures']:
    #         weight = mixture['weight']
    #         mean = mixture['mean']
    #         cov = mixture['cov']
            
    #         # Calculate multivariate Gaussian probability
    #         prob = multivariate_normal.pdf(observation, mean=mean, cov=cov)
            
    #         total_prob += weight * prob
        
    #     return np.log(total_prob)  # Return log probability

    def emission_prob(self, state, observation):
        if isinstance(state, int):
            # Handle state as an integer index
            state_info = None
            for hmm_name, hmm in self.hmms.items():
                if state < len(hmm['states']):
                    state_info = hmm['states'][state]
                    break
            if state_info is None:
                # raise ValueError(f"State index {state} not found in any HMM")
                return 0
                
        else:
            # Handle state as a string (original implementation)
            hmm_name, state_num = state.split('.')
            hmm = self.hmms[hmm_name]
            state_info = hmm['states'][int(state_num)]
        
        # Calculate univariate Gaussian probability
        # print('state_info ',state_info)
        k1 = 'mean'
        k2 = 'variance'
        if k1 in state_info.keys() and k2 in state_info.keys():
            mean = state_info['mean']
            variance = state_info['variance']

            obs = observation[:len(mean)] if len(observation) > len(mean) else np.pad(observation, (0, len(mean) - len(observation)))
            
            diff = obs - mean
            exponent = -0.5 * np.sum((diff ** 2) / variance)
            det = np.prod(variance)
            norm_const = 1.0 / np.sqrt((2 * np.pi) ** len(mean) * det)
            
            prob = norm_const * np.exp(exponent)
        else:
            prob=1
        
        return np.log(prob)  # Return log probability


    def transition_prob(self, from_state, to_state, hmm_name):
        return self.transition_matrices[hmm_name][from_state][to_state]

    def get_num_states(self, hmm_name):
        return len(self.hmms[hmm_name]['states'])

    def get_initial_prob(self, hmm_name):
        # Assuming the first state is always the initial state
        return self.transition_matrices[hmm_name][0]

    def get_final_prob(self, hmm_name):
        # Assuming the last state is always the final state
        last_state = self.get_num_states(hmm_name) - 1
        return self.transition_matrices[hmm_name][last_state][-1]
    
    def load_macros(self, macros_file):
        with open(macros_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('~'):
                macro_type, macro_name = line[1], line[3:]
                if macro_type == 'o':
                    # Load options
                    self.load_options(lines[i+1:])
                elif macro_type == 'v':
                    # Load variance definition
                    self.load_variance_macro(macro_name, lines[i+1:])
            i += 1
    
    def load_options(self, lines):
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('<'):
                break
            parts = line.split('=')
            if len(parts) == 2:
                option, value = parts[0].strip(), parts[1].strip()
                if option == 'STREAMINFO':
                    self.swidth = [int(x) for x in value.split()]
                elif option == 'VECSIZE':
                    self.vecSize = int(value)
                elif option == 'MSDINFO':
                    self.msdflag = [int(x) for x in value.split()]
                # Add more options as needed
            i += 1
        return i

    def load_variance_macro(self, name, lines):
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('<VARIANCE>'):
                # Extract the variance values directly from this line
                variance_values = line.split()[1:]  # Skip the '<VARIANCE>' tag
                variance = [float(x) for x in variance_values]
                self.variance_macros[name] = np.array(variance)
                return i + 1  # Move to the next line after processing
            i += 1
        return i
    
    def load_hmmdef(self, hmmdef_file):
        with open(hmmdef_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('~h'):
                hmm_name = line.split('"')[1]
                print(f"Processing HMM: {hmm_name}")
                i = self.parse_hmm(hmm_name, lines, i + 1)
                self.numPhyHMM += 1
                self.numLogHMM += 1
            else:
                i += 1
        
        print(f"Loaded {self.numPhyHMM} HMMs")



    def parse_hmm(self, hmm_name, lines, start):
        num_states = 0
        states = []
        transitions = None
        max_iterations = 1000  # Add a maximum iteration limit

        i = start
        iteration = 0
        while i < len(lines) and iteration < max_iterations:
            line = lines[i].strip()
            # print("This is the line: ",line)
            if line.startswith('<NUMSTATES>'):
                num_states = int(line.split()[1])
                states = [{'mixtures': []} for _ in range(num_states)]
                i += 1
            elif line.startswith('<STATE>'):
                state_idx = int(line.split()[1]) - 1
                i = self.parse_state(states[state_idx], lines, i + 1)
            elif line.startswith('<TRANSP>'):
                transitions, i = self.parse_transitions(lines, i + 1, num_states)
            elif line.startswith('<ENDHMM>'):
                self.add_hmm(hmm_name, states, transitions, [])
                return i + 1
            else:
                i += 1
            iteration += 1

        if iteration >= max_iterations:
            print(f"Warning: Maximum iterations reached while parsing HMM {hmm_name}")
        return i


    def parse_state(self, state, lines, start):
        i = start
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('<MEAN>'):
                size = int(line.split()[1])
                i += 1
                mean = [float(x) for x in lines[i].split()]
                if len(mean) != size:
                    raise ValueError(f"Expected {size} values for MEAN, got {len(mean)}")
                state['mean'] = mean
            elif line.startswith('<VARIANCE>'):
                size = int(line.split()[1])
                i += 1
                variance = [float(x) for x in lines[i].split()]
                if len(variance) != size:
                    raise ValueError(f"Expected {size} values for VARIANCE, got {len(variance)}")
                state['variance'] = variance
            elif line.startswith('<GCONST>'):
                state['gconst'] = float(line.split()[1])
            elif line.startswith('<STATE>') or line.startswith('<TRANSP>'):
                return i  # Return to parse_hmm function
            i += 1
        return i


    def parse_mixture(self, state, lines, start):
        mixture = {'weight': 1.0, 'mean': None, 'cov': None}
        i = start
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('<MIXTURE>'):
                mixture['weight'] = float(line.split()[2])
            elif line.startswith('<MEAN>'):
                mixture['mean'] = [float(x) for x in lines[i+1].strip().split()]
                i += 1
            elif line.startswith('<VARIANCE>'):
                mixture['cov'] = [float(x) for x in lines[i+1].strip().split()]
                i += 1
                state['mixtures'].append(mixture)
                return i + 1
            i += 1
        return i

    def parse_transitions(self, lines, start, num_states):
        transitions = []
        i = start
        for _ in range(num_states):
            line = lines[i].strip()
            if line.startswith('<ENDHMM>'):
                break
            transitions.append([float(x) for x in line.split()])
            i += 1
        return transitions, i

    
    def attach_precomps(self):
        for hmm in self.hmms.values():
            for state in hmm['states']:
                for mixture in state['mixtures']:
                    self.precompute_gaussian_constants(mixture)

    def precompute_gaussian_constants(self, mixture):
        dim = len(mixture['mean'])
        det = np.prod(mixture['cov'])
        mixture['gconst'] = -0.5 * (dim * np.log(2 * np.pi) + np.log(det))
        mixture['inv_cov'] = 1.0 / mixture['cov']

class ViterbiRecognizer:
    def __init__(self, hmm_set, vocabulary, network, lm_scale=1.0, word_penalty=0.0):
        self.hmm_set = hmm_set
        self.vocabulary = vocabulary
        self.network = None
        self.lm_scale = lm_scale
        self.word_penalty = word_penalty
    
    def set_network(self, network):
        self.network = network

    def recognize(self, observations):
        num_frames = len(observations)
        num_states = self.network.num_states
        # print("num_frames ",num_frames)
        # Initialize Viterbi variables
        viterbi = np.zeros((num_frames, num_states))
        backpointers = np.zeros((num_frames, num_states), dtype=int)
        
        # Initialize first frame
        for s in range(num_states):
            emProb = 0
            if num_frames>0:
                emProb = self.hmm_set.emission_prob(s, observations[0])
            else:
                emProb = 0
            viterbi[0, s] = self.network.initial_prob(s) + emProb
        
        # Viterbi recursion
        for t in range(1, num_frames):
            for s in range(num_states):
                max_prob = float('-inf')
                max_state = -1
                
                for prev_s in self.network.prev_states(s):
                    prob = (viterbi[t-1, prev_s] + 
                            self.network.transition_prob(prev_s, s) * self.lm_scale + 
                            self.hmm_set.emission_prob(s, observations[t]))
                    
                    if prob > max_prob:
                        max_prob = prob
                        max_state = prev_s
                
                viterbi[t, s] = max_prob
                backpointers[t, s] = max_state
        
        # Termination
        final_state = np.argmax(viterbi[-1])
        best_path_prob = viterbi[-1, final_state]
        
        # Backtracking
        best_path = [final_state]
        for t in range(num_frames - 1, 0, -1):
            best_path.insert(0, backpointers[t, best_path[0]])
        
        # Convert state sequence to word sequence
        word_sequence = self.state_to_word_sequence(best_path)
        
        return word_sequence, best_path_prob
    
    def state_to_word_sequence(self, state_sequence):
        word_sequence = []
        current_word = None
        
        for state in state_sequence:
            word = self.network.state_to_word(state)
            if word != current_word:
                if word is not None:
                    word_sequence.append(word)
                current_word = word
        
        return word_sequence

# def extract_mfcc(file_path, num_mfcc=13, n_fft=2048, hop_length=512):
#     # Load the audio file
#     y, sr = librosa.load(file_path, sr=None)
    
#     # Extract MFCCs
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    
#     # Normalize MFCCs
#     mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)
    
#     return mfccs.T
# def load_observations(file_path):
#     # print("FILE PATH ",file_path)

#     with open(file_path, 'rb') as f:
#         # Read header
#         nSamples = np.frombuffer(f.read(4), dtype=np.int32)[0]
#         sampPeriod = np.frombuffer(f.read(4), dtype=np.int32)[0]
#         sampSize = np.frombuffer(f.read(2), dtype=np.int16)[0]
#         parmKind = np.frombuffer(f.read(2), dtype=np.int16)[0]

        
        
#         # Read data
#         raw_data = f.read()
#         # Calculate padding
#         padding_size = (4 - len(raw_data) % 4) % 4
        
#         # Pad the raw_data
#         padded_data = np.pad(np.frombuffer(raw_data, dtype=np.uint8), (0, padding_size), 'constant')
#         data = np.frombuffer(padded_data, dtype=np.float32)
        
#         print("DATA Len",len(data))
#         # Calculate the actual number of complete samples
#         actual_samples = len(data) // sampSize
        
#         if actual_samples != nSamples:
#             print(f"Warning: Expected {nSamples} samples, but found {actual_samples} complete samples.")
        
#         # Reshape the data, using only complete samples
#         observations = data[:actual_samples * sampSize].reshape(-1, sampSize)
    
#     return observations

# def load_observations(file_path):
#     # Load the audio file
#     y, sr = librosa.load(file_path, sr=None)
    
#     # Extract MFCC features
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
#     # Transpose to get time as the first dimension
#     mfcc = mfcc.T
    
#     # Normalize the MFCC features
#     mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
    
#     return mfcc
def load_observations(file_path):
    htk_file = HTKFile()
    htk_file.load(file_path)
    
    observations = np.array(htk_file.data)
    
    # Ensure observations is a 2D numpy array
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)
    
    return observations

# def load_observations(file_path):
#     # Load the audio file
#     audio = AudioSegment.from_file(file_path)
    
#     # Convert to numpy array
#     samples = np.array(audio.get_array_of_samples())
    
#     # Normalize to float between -1 and 1
#     samples = samples.astype(np.float32) / 32768.0
    
#     # If stereo, convert to mono
#     if audio.channels == 2:
#         samples = samples.reshape((-1, 2)).mean(axis=1)
    
#     # Extract MFCC features (you may need to implement this part)
#     mfcc = extract_mfcc(samples, audio.frame_rate)
    
#     return mfcc  # or return mfcc if you implement MFCC extraction

def parse_config(config_file):
    config = {
        'lm_scale': 1.0,  # Default value
        'word_penalty': 0.0  # Default value
    }
    with open(config_file, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=')
                key = key.strip()
                value = value.strip()
                if key == 'lm_scale' or key == 'word_penalty':
                    config[key] = float(value)
                else:
                    config[key] = value
    return config

def load_hmm_set(hmmdef_file, macros_file):
    hmm_set = HMMSet()
    
    # Load macros file
    hmm_set.load_macros(macros_file)
    
    # Load HMM definitions
    hmm_set.load_hmmdef(hmmdef_file)
    
    # Perform necessary initializations
    hmm_set.attach_precomps()
    
    # Set up adaptation-related structures if needed
    # if hmm_set.adaptation_enabled:
    #     hmm_set.create_adapt_xform("tmp")
    
    return hmm_set

def read_mlf(mlf_file):
    mlf_data = {}
    current_file = None
    current_labels = []
    
    with open(mlf_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '#!MLF!#':
                continue
            elif line.startswith('"'):
                if current_file is not None:
                    mlf_data[current_file] = current_labels
                current_file = line.strip('"')
                current_labels = []
            elif line == '.':
                if current_file is not None:
                    mlf_data[current_file] = current_labels
                    current_file = None
                    current_labels = []
            else:
                # Append each label directly to the current labels list
                current_labels.append(line)
    
    # Handle the last file case if it doesn't end with a '.'
    if current_file is not None and current_labels:
        mlf_data[current_file] = current_labels
    
    return mlf_data


def write_aligned_mlf(aligned_data, output_mlf):
    with open(output_mlf, 'w') as f:
        f.write('#!MLF!#\n')
        for file_path, labels in aligned_data.items():
            f.write(f'"{file_path}"\n')
            for label in labels:
                f.write(f'{label}\n')
            f.write('.\n')

def read_scp(scp_file):
    mfcc_files = []
    with open(scp_file, 'r') as f:
        for line in f:
            mfcc_files.append(line.strip())
    return mfcc_files

def load_dictionary(dict_file):
    vocabulary = {}
    with open(dict_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                pronunciation = parts[1:]
                if word not in vocabulary:
                    vocabulary[word] = []
                vocabulary[word].append(pronunciation)
    return vocabulary

def load_monophones(monophones_file):
    with open(monophones_file, 'r') as f:
        return [line.strip() for line in f]


def main():
    parser = argparse.ArgumentParser(description="HMM-based forced alignment")
    parser.add_argument("--config", required=True, help="Configuration file")
    parser.add_argument("--macros", required=True, help="Macros file")
    parser.add_argument("--hmmdef", required=True, help="HMM definition file")
    parser.add_argument("--mlf", required=True, help="Words MLF file")
    parser.add_argument("--scp", required=True, help="Training SCP file")
    parser.add_argument("--dict", required=True, help="Dictionary file")
    parser.add_argument("--monophones", required=True, help="Monophones file")
    parser.add_argument("--output", required=True, help="Output aligned MLF file")

    args = parser.parse_args()

    config = parse_config(args.config)
    hmm_set = load_hmm_set(args.hmmdef, args.macros)
    # print("hahahaha ",hmm_set.hmms)
    word_labels = read_mlf(args.mlf)  # Load labels from MLF
    mfcc_files = read_scp(args.scp)  # Read MFCC files from SCP
    dictionary = load_dictionary(args.dict)
    monophones = load_monophones(args.monophones)

    viterbi_recognizer = ViterbiRecognizer(hmm_set, dictionary, config['lm_scale'], config['word_penalty'])

    aligned_data = {}
    # print(word_labels)
    for mfcc_file in mfcc_files:
        observations = load_observations(mfcc_file)
        # print("OBSERVATIONS! ",len(observations))
        # Extract the corresponding label list from word_labels using the current mfcc_file
        # Assuming that the MLF contains paths that match the MFCC files
        if len(observations)==0:
            continue
        label_key = '*'+mfcc_file[20:-4] + '.lab'  # Format it to match MLF key format
        # print("hehehe!",label_key)
        if label_key in word_labels:
            words = word_labels[label_key]  # Get labels for this MFCC file
            
            # Process words into phone sequence if needed
            phone_sequence = []
            for word in words:
                if word in dictionary:
                    phone_sequence.extend(dictionary[word])
                else:
                    print(f"Warning: Word '{word}' not found in dictionary")

            # Create network for this utterance
            network = create_network_from_words(words, hmm_set, dictionary)
        
            # Set the network for this utterance
            viterbi_recognizer.set_network(network)
            
            alignment, _ = viterbi_recognizer.recognize(observations)  # Pass observations for recognition
            print("hello!!!!!! ",alignment)
            aligned_data[mfcc_file] = alignment  # Store alignment results
        else:
            print(f"Warning: No labels found for MFCC file {mfcc_file}")

    # print(aligned_data)
    write_aligned_mlf(aligned_data, args.output)  # Write aligned data to output MLF

    print(f"Alignment complete. Output written to {args.output}")

if __name__ == "__main__":
    main()
