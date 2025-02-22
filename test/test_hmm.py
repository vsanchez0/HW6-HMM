import pytest
from hmm import HiddenMarkovModel
import numpy as np


def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """
    mini_hmm=np.load('./data/mini_weather_hmm.npz', allow_pickle=True)
    hidden_states = mini_hmm['hidden_states']
    observation_states = mini_hmm['observation_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']

    mini_input=np.load('./data/mini_weather_sequences.npz', allow_pickle=True)
    observation_state_sequence = mini_input['observation_state_sequence']
    expected_viterbi_sequence = mini_input['best_hidden_state_sequence']

    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    forward_probability = hmm.forward(observation_state_sequence)
    assert isinstance(forward_probability, float), "Forward algorithm should return a float probability."
    assert forward_probability > 0, "Forward probability should be positive."
    
    predicted_viterbi_sequence = hmm.viterbi(observation_state_sequence)
    assert len(predicted_viterbi_sequence) == len(expected_viterbi_sequence), "Viterbi sequence length mismatch."
    assert all(isinstance(state, str) for state in predicted_viterbi_sequence), "Viterbi output states should be strings."
    assert predicted_viterbi_sequence == list(expected_viterbi_sequence), "Viterbi sequence does not match expected sequence."
    
    empty_sequence = np.array([])
    try:
        empty_viterbi_output = hmm.viterbi(empty_sequence)
        assert empty_viterbi_output == [], "Empty sequence should return an empty list."
    except Exception as e:
        assert False, f"Error handling empty sequence: {e}"
    
    unseen_sequence = np.array(["hail", "fog", "tornado"])
    try:
        hmm.viterbi(unseen_sequence)
        assert False, "Viterbi should raise an error for unseen observation states."
    except ValueError as e:
        assert "Unknown observation state" in str(e), f"Unexpected error message: {e}"
    
    print("All mini_weather HMM tests passed!")

def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm=np.load('./data/full_weather_hmm.npz', allow_pickle=True)
    hidden_states = full_hmm['hidden_states']
    observation_states = full_hmm['observation_states']
    prior_p = full_hmm['prior_p']
    transition_p = full_hmm['transition_p']
    emission_p = full_hmm['emission_p']

    full_input=np.load('./data/full_weather_sequences.npz', allow_pickle=True)
    observation_state_sequence = full_input['observation_state_sequence']
    expected_viterbi_sequence = full_input['best_hidden_state_sequence']

    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    forward_probability = hmm.forward(observation_state_sequence)
    assert isinstance(forward_probability, float), "Forward algorithm should return a float probability."
    assert forward_probability > 0, "Forward probability should be positive."
    
    predicted_viterbi_sequence = hmm.viterbi(observation_state_sequence)
    assert len(predicted_viterbi_sequence) == len(expected_viterbi_sequence), "Viterbi sequence length mismatch."
    assert all(isinstance(state, str) for state in predicted_viterbi_sequence), "Viterbi output states should be strings."
    assert predicted_viterbi_sequence == list(expected_viterbi_sequence), "Viterbi sequence does not match expected sequence."
    
    print("All full_weather HMM tests passed!")

test_mini_weather()
test_full_weather()












