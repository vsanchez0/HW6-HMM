import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """      
        M = len(input_observation_states)
        N = len(self.hidden_states)
        # N = self.prior_p.shape[0]
        alpha = np.zeros((M, N))

        alpha[0,:] = self.prior_p * self.emission_p[:, self.observation_states_dict[input_observation_states[0]]]

        for t in range(1,M):
            for j in range(N):
        #         for i in range(N):
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.transition_p[:, j]) * self.emission_p[j, self.observation_states_dict[input_observation_states[t]]]
        #             alpha[t,j]+= alpha[t-1, i] * self.transition_p[i, j] * self.emission_p[j, self.input_observation_states[t]]

        return np.sum(alpha[M-1,:])


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        

        M = len(decode_observation_states)
        N = len(self.hidden_states)

        if M == 0:
            return []

        viterbi_table = np.zeros((M, N))
        backpointer = np.zeros((M, N), dtype=int)

        obs_index = self.observation_states_dict.get(decode_observation_states[0], None)
        if obs_index is None:
            raise ValueError(f"Unknown observation state: {decode_observation_states[0]}")

        for j in range(N):
            viterbi_table[0, j] = self.prior_p[j] * self.emission_p[j, obs_index]

        for t in range(1, M):
            obs_index = self.observation_states_dict.get(decode_observation_states[t], None)
            if obs_index is None:
                raise ValueError(f"Unknown observation state: {decode_observation_states[t]}")

            for j in range(N):
                max_prob, best_state = max(
                    (viterbi_table[t - 1, i] * self.transition_p[i, j], i) for i in range(N)
                )
                viterbi_table[t, j] = max_prob * self.emission_p[j, obs_index]
                backpointer[t, j] = best_state

        best_final_state = np.argmax(viterbi_table[M - 1, :])
        best_path = np.zeros(M, dtype=int)
        best_path[-1] = best_final_state

        for t in range(M - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        best_hidden_state_sequence = [self.hidden_states[i] for i in best_path]
        return best_hidden_state_sequence