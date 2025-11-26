import numpy as np 

def compute_softmax_prob(actor_w, tiles):
    """
    Computes softmax probability for all actions
    
    Args:
    actor_w - np.array, an array of actor weights
    tiles - np.array, an array of active tiles
    
    Returns:
    softmax_prob - np.array, an array of size equal to num. actions, and sums to 1.
    """
    
    # First compute the list of state-action preferences (1~2 lines)
    # state_action_preferences = ? (list of size 3)
    state_action_preferences = []
    # ----------------
    # your code here
    
    # print(actor_w.shape)
    for c in actor_w:
        #print("###########")
        #print(c)
        action_pref = c[tiles].sum()
        state_action_preferences.append(action_pref)
    # print("State Pref")
    # print(state_action_preferences)
    # for i in range(actor_w):
        
    
    # ----------------
    
    # Set the constant c by finding the maximum of state-action preferences (use np.max) (1 line)
    # c = ? (float)
    # ----------------
    # your code here
    c = np.max(state_action_preferences)
    
    # ----------------
    
    # Compute the numerator by subtracting c from state-action preferences and exponentiating it (use np.exp) (1 line)
    # numerator = ? (list of size 3)
    # ----------------
    # your code here
    numerator = np.exp(state_action_preferences - c)
    
    # ----------------
    
    # Next compute the denominator by summing the values in the numerator (use np.sum) (1 line)
    # denominator = ? (float)
    # ----------------
    # your code here
    denominator = np.sum(numerator)
    
    # ----------------
    
    
    # Create a probability array by dividing each element in numerator array by denominator (1 line)
    # We will store this probability array in self.softmax_prob as it will be useful later when updating the Actor
    # softmax_prob = ? (list of size 3)
    # ----------------
    # your code here
    softmax_prob = numerator / denominator
    # self.softmax_prob = softmax_prob
    
    # ----------------
    
    return softmax_prob