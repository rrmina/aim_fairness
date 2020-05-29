import numpy as np
# from tabulate import tabulate 

def report_group_metrics(A, Y_hat, Y):
    assert Y_hat.shape == Y.shape, "Y_hat and Y are expected to have the same shape"
    assert Y_hat.shape == A.shape, "Y_hat and A are expected to have the same shape"
    assert len(A.shape) == 1, "A, Y_hat, and Y are expected to be 1D vectors"

    # Compute metrics
    base_rate = Base_rate(A, Y)
    dp_gap = DP_gap(A, Y_hat)
    eo_gap = EO_gap(A, Y_hat, Y)
    eopp_gap = EOpp_gap(A, Y_hat, Y)
    acc_gap = Acc_gap(A, Y_hat, Y)

    # Pretty Printing
    headers = ["Fairness Definitions", "Gap Values"]
    row_format = "{:>30}" * len(headers)
    data = []
    data.append(["Base Rate", str(base_rate)])
    data.append(["Demographic Parity", str(dp_gap)])
    data.append(["Equalized Odds", str(eo_gap)])
    data.append(["Equal Opportunity Y0", str(eopp_gap[0])])
    data.append(["Equal Opportunity Y1", str(eopp_gap[1])])
    data.append(["Accuracy Parity", str(acc_gap)])

    # Print
    print(row_format.format(*headers))
    print("=" * 70)
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
        print("Pos Rate A0: {} || Pos Rate A1: {}".format(PosRate0, PosRate1))

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
        print("Eopp_Y0: {} || PosRate_A0_Y0: {} || PosRate_A1_Y0: {}".format(Eopp_Y0, PosRate_A0_Y0, PosRate_A1_Y0))
        print("Eopp_Y1: {} || PosRate_A0_Y1: {} || PosRate_A1_Y1: {}".format(Eopp_Y1, PosRate_A0_Y1, PosRate_A1_Y1))

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
        print("Eopp_Y0: {} || PosRate_A0_Y0: {} || PosRate_A1_Y0: {}".format(Eopp_Y0, PosRate_A0_Y0, PosRate_A1_Y0))
        print("Eopp_Y1: {} || PosRate_A0_Y1: {} || PosRate_A1_Y1: {}".format(Eopp_Y1, PosRate_A0_Y1, PosRate_A1_Y1))

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
        print("Acc A0: {} || Acc A1: {}".format(Acc0, Acc1))

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
        print("Base Rate A0: {} || Base Rate A1: {}".format(BaseRate0, BaseRate1))

    return abs(BaseRate0 - BaseRate1) 