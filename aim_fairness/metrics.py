import numpy as np

def report_group_metrics(A, Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    assert Y_hat.shape == A.shape, "Y_hat and A are expected to have the same shape"
    assert len(A.shape) == 1, "A, Y_hat, and Y are expected to be 1D vectors"

    if (debug):
        print("=" * 70)
        print("{}{}".format(" " * 30, "Debug Logs"))
        print("=" * 70)

    # Compute metrics
    base_rate = Base_rate(A, Y, debug)
    dp_gap = DP_gap(A, Y_hat, debug)
    eo_gap = EO_gap(A, Y_hat, Y, debug)
    eopp_gap = EOpp_gap(A, Y_hat, Y, debug)
    acc_gap = Acc_gap(A, Y_hat, Y, debug)

    # Pretty Printing    
    row_format = "{:>30}{:>30.4f}"
    data = []
    data.append(["Base Rate", base_rate])
    data.append(["Demographic Parity", dp_gap])
    data.append(["Equalized Odds", eo_gap])
    data.append(["Equal Opportunity Y0", eopp_gap[0]])
    data.append(["Equal Opportunity Y1", eopp_gap[1]])
    data.append(["Accuracy Parity", acc_gap])

    # Print
    headers = ["Fairness Definitions", "Gap Values"]
    print("=" * (len(headers) * 30 + 10))
    print("{:>30}{:>30}".format(*headers))
    print("=" * (len(headers) * 30 + 10))
    for row in data:
        print(row_format.format(*row))

def DP_gap(A, Y_hat, debug=False):
    assert Y_hat.shape == A.shape, "Y_hat and A are expected to have the same shape"
    assert len(A.shape) == 1, "A and Y are expected to be 1D vectors"

    # Group 0 Predictions
    Y_hat_A0 = Y_hat[A == 0]

    # Group 1 Prediction
    Y_hat_A1 = Y_hat[A == 1]

    # Get the positive rates
    PosRate0 = np.mean((Y_hat_A0 == 1) * 1)
    PosRate1 = np.mean((Y_hat_A1== 1) * 1)

    if (debug):
        print("Demographic Parity")
        print("{:>5} Positive Rate (A=0): {:>20.4f}".format("", PosRate0))
        print("{:>5} Positive Rate (A=1): {:>20.4f}".format("", PosRate1))

    return abs(PosRate0 - PosRate1) 

def EO_gap(A, Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    assert Y_hat.shape == A.shape, "Y_hat and A are expected to have the same shape"
    assert len(A.shape) == 1, "A, Y_hat, and Y are expected to be 1D vectors"

    Y_hat_A0 = Y_hat[A==0]      # Predictions of Group 0
    Y_hat_A1 = Y_hat[A==1]      # Predictions of Group 1
    Y_A0 = Y[A==0]              # Labels of Group 0
    Y_A1 = Y[A==1]              # Labels of Group 1

    Y_hat_A0_Y1 = Y_hat_A0[Y_A0==1]           # Predictions of Group 0 with corresponding Ground Truth Labels 1
    Y_hat_A1_Y1 = Y_hat_A1[Y_A1==1]           # Predictions of Group 1 with corresponding Ground Truth Labels 1
    Y_hat_A0_Y0 = Y_hat_A0[Y_A0==0]           # Predictions of Group 0 with corresponding Ground Truth Labels 0
    Y_hat_A1_Y0 = Y_hat_A1[Y_A1==0]           # Predictions of Group 1 with corresponding Ground Truth Labels 0

    # Positive Rates given the conditions
    PosRate_A0_Y1 = np.mean((Y_hat_A0_Y1 == 1) * 1)
    PosRate_A1_Y1 = np.mean((Y_hat_A1_Y1 == 1) * 1)
    PosRate_A0_Y0 = np.mean((Y_hat_A0_Y0 == 1) * 1)
    PosRate_A1_Y0 = np.mean((Y_hat_A1_Y0 == 1) * 1)

    # Equal Opportunity on both sides
    Eopp_Y0 = abs(PosRate_A0_Y0 - PosRate_A1_Y0)
    Eopp_Y1 = abs(PosRate_A0_Y1 - PosRate_A1_Y1)

    if (debug):
        print("Equalized Odds")
        print("{:>5} Equal Opporunity (Y=0)".format(""))
        print("{:>5} {:>5} Positive Rate (A=0) and (Y=0): {:>20.4f}".format("", "", PosRate_A0_Y0))
        print("{:>5} {:>5} Positive Rate (A=1) and (Y=0): {:>20.4f}".format("", "", PosRate_A1_Y0))
        print("{:>5} Equal Opporunity (Y=1)".format(""))
        print("{:>5} {:>5} Positive Rate (A=0) and (Y=1): {:>20.4f}".format("", "", PosRate_A0_Y1))
        print("{:>5} {:>5} Positive Rate (A=1) and (Y=1): {:>20.4f}".format("", "", PosRate_A1_Y1))

    return Eopp_Y0 + Eopp_Y1

def EOpp_gap(A, Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    assert Y_hat.shape == A.shape, "Y_hat and A are expected to have the same shape"
    assert len(A.shape) == 1, "A, Y_hat, and Y are expected to be 1D vectors"

    Y_hat_A0 = Y_hat[A==0]      # Predictions of Group 0
    Y_hat_A1 = Y_hat[A==1]      # Predictions of Group 1
    Y_A0 = Y[A==0]              # Labels of Group 0
    Y_A1 = Y[A==1]              # Labels of Group 1

    Y_hat_A0_Y1 = Y_hat_A0[Y_A0==1]           # Predictions of Group 0 with corresponding Ground Truth Labels 1
    Y_hat_A1_Y1 = Y_hat_A1[Y_A1==1]           # Predictions of Group 1 with corresponding Ground Truth Labels 1
    Y_hat_A0_Y0 = Y_hat_A0[Y_A0==0]           # Predictions of Group 0 with corresponding Ground Truth Labels 0
    Y_hat_A1_Y0 = Y_hat_A1[Y_A1==0]           # Predictions of Group 1 with corresponding Ground Truth Labels 0

    # Positive Rates given the conditions
    PosRate_A0_Y1 = np.mean((Y_hat_A0_Y1 == 1) * 1)
    PosRate_A1_Y1 = np.mean((Y_hat_A1_Y1 == 1) * 1)
    PosRate_A0_Y0 = np.mean((Y_hat_A0_Y0 == 1) * 1)
    PosRate_A1_Y0 = np.mean((Y_hat_A1_Y0 == 1) * 1)

    # Equal Opportunity on both sides
    Eopp_Y0 = abs(PosRate_A0_Y0 - PosRate_A1_Y0)
    Eopp_Y1 = abs(PosRate_A0_Y1 - PosRate_A1_Y1)

    if (debug):
        print("Equal Opportunity")
        print("{:>5} Equal Opporunity (Y=0)".format(""))
        print("{:>5} {:>5} Positive Rate (A=0) and (Y=0): {:>20.4f}".format("", "", PosRate_A0_Y0))
        print("{:>5} {:>5} Positive Rate (A=1) and (Y=0): {:>20.4f}".format("", "", PosRate_A1_Y0))
        print("{:>5} Equal Opporunity (Y=1)".format(""))
        print("{:>5} {:>5} Positive Rate (A=0) and (Y=1): {:>20.4f}".format("", "", PosRate_A0_Y1))
        print("{:>5} {:>5} Positive Rate (A=1) and (Y=1): {:>20.4f}".format("", "", PosRate_A1_Y1))

    return Eopp_Y0, Eopp_Y1

def Acc_gap(A, Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    assert Y_hat.shape == A.shape, "Y_hat and A are expected to have the same shape"
    assert len(A.shape) == 1, "A, Y_hat, and Y are expected to be 1D vectors"

    # Group 0 Ground Truth and Predictions
    Y_A0 = Y[A==0]
    Y_hat_A0 = Y_hat[A==0]

    # Group 1 Ground Truth and Predictions
    Y_A1 = Y[A==1]
    Y_hat_A1 = Y_hat[A==1]

    # Get the accuracies for both groups
    Acc0 = np.mean((Y_A0 == Y_hat_A0)*1)
    Acc1 = np.mean((Y_A1 == Y_hat_A1)*1)
    
    if (debug):
        print("Accuracy Parity")
        print("{:>5} Accuracy (A=0): {:>20.4f}".format("", Acc0))
        print("{:>5} Accuracy (A=1): {:>20.4f}".format("", Acc1))

    return abs(Acc0 - Acc1)

# DP_gap and Base are essentially the same, except for their inputs
def Base_rate(A, Y, debug=False):
    assert Y.shape == A.shape, "Y and A are expected to have the same shape"
    assert len(A.shape) == 1, "A and Y are expected to be 1D vectors"

    # Group 0 Labels
    Y_A0 = Y[A == 0]

    # Group 1 Labels
    Y_A1 = Y[A == 1]

    # Get the positive rates
    BaseRate0 = np.mean((Y_A0 == 1) * 1)
    BaseRate1 = np.mean((Y_A1== 1) * 1)

    if (debug):
        print("Base Rate")
        print("{:>5} Base Rate (A=0): {:>20.4f}".format("", BaseRate0))
        print("{:>5} Base Rate (A=1): {:>20.4f}".format("", BaseRate1))

    return abs(BaseRate0 - BaseRate1) 

def F1_score_gap(A, Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    assert Y_hat.shape == A.shape, "Y_hat and A are expected to have the same shape"
    assert len(A.shape) == 1, "A, Y_hat, and Y are expected to be 1D vectors"

    return F_beta_score_gap(A, Y_hat, Y, beta=1, debug=debug)

def F_beta_score_gap(A, Y_hat, Y, beta=1, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    assert Y_hat.shape == A.shape, "Y_hat and A are expected to have the same shape"
    assert len(A.shape) == 1, "A, Y_hat, and Y are expected to be 1D vectors"

    Y_A0 = Y[A==0]
    Y_A1 = Y[A==1]
    Y_hat_A0 = Y_hat[A==0]
    Y_hat_A1 = Y_hat[A==1]

    F_beta_score_A0 = F_beta_score(Y_hat_A0, Y_A0, debug=debug)
    F_beta_score_A1 = F_beta_score(Y_hat_A1, Y_A1, debug=debug)

    if (debug):
        print("F{} Scores".format(beta))
        print("{:>5} F{} Score (A=0): {:>20.4f}".format("", beta, F_beta_score_A0))
        print("{:>5} F{} Score (A=1): {:>20.4f}".format("", beta, F_beta_score_A1))

    return abs(F_beta_score_A0 - F_beta_score_A1)

def F1_score(Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    return F_beta_score(Y_hat, Y, beta=1, debug=debug)

def F_beta_score(Y_hat, Y, beta=1, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    precision_ = precision(Y_hat, Y)
    recall_ = recall(Y_hat, Y)

    if (debug):
        print("{:>5} Precision: {:>20.4f}".format("", precision_))
        print("{:>5} Recall: {:>20.4f}".format("", recall_))

    constant = (1+ np.power(beta, 2))
    numerator = precision_ * recall_
    denominator = (np.power(beta, 2) * precision_) + recall_

    return constant * numerator / denominator

# Sensitivty, Recall, Hit Rate or True Positive Rate (TPR)
def recall(Y_hat, Y):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    return np.mean(Y_hat[Y==1])

# Specificity, Selectivity, or True Negative Rate (TNR)
def specificity(Y_hat, Y):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    return 1 - np.mean(Y_hat[Y==0])

# Precision or Positive Prediction Value
def precision(Y_hat, Y):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    return np.mean(Y[Y_hat==1])