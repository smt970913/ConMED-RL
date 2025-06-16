"""
Examples of custom done condition functions for different decision-making scenarios.
These functions can be passed to the data_buffer_train and data_buffer_val methods.
"""

def discharge_done_condition(action, outcome_df, idx, **kwargs):
    """
    The discharge decision-making logic.
    
    Args:
        action: The action taken (0 or 1 for discharge decision)
        outcome_df: pandas DataFrame containing outcome data for current timestep
        idx: index of the current state
        kwargs: additional keyword arguments
    Returns:
        float: 1.0 if episode should end, 0.0 if continue
    """
    if action == 1.0:
        if outcome_df['death'].values[idx] == 1.0:
            if outcome_df['discharge_fail'].values[idx] == 1.0:
                return 0.0
            else:
                return 1.0
        else:
            if outcome_df['discharge_fail'].values[idx] == 0.0:
                return 1.0
            else:
                return 0.0
    else:
        return 0.0

