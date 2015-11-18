from options import *
from functions import *

def main(userId, option, option2):
    """
    If a userId was provided, it will output the top 5 recommendations using the model specified in the options.

    Otherwise, it will evaluate the chosen recommendation algorithm.
    """

    df = load_purchase_history()

    if userId:
        topK_pIds, pId_int = topK_for_uId(userId, df)
        print_topK_pIds(topK_pIds, pId_int)

    else:
        eval(df)


def topK_for_uId(uId, df):
    """
    Calculates the top5 product recommendations depending on user segmentation and model predictions.

    Parameters
    ----------
    uId: user Id, integer
    df: dataframe from purchase history (see function load_purchase_history())

    Returns
    -------
    topK_pIds: product Ids of the top 5 recommendations
    pId_int: dictionary of productId<->integer equivalences, created by function prepare_df
    """

    # for the baseline, we return the all-time top5 best-sellers (removing previous user purchases)
    if option2=='baseline':
        df, pId_int = prepare_df(df, special='baseline')
        return topK_no_coll_filt(df, None, uId, 'baseline'), pId_int

    df, pId_int, uIds_outliers, df_outliers, nothing  = prepare_df(df)

    u_segment = classify_user(uId, df, uIds_outliers)

    # in these cases we don't need to train a collaborative filtering model, we can just directly recommend:
    if u_segment=='new_user':
        return topK_no_coll_filt(df, None, uId, u_segment), pId_int
    elif u_segment=='outlier' or option2=='only_custom_BS':
        return topK_no_coll_filt(df, df_outliers, uId, u_segment), pId_int

    # else, we must train a model to complement our recommendations:
    else:
        model, model_name = train_recommender(df, option)
        lists_of_bestsellers = calc_ntopK_best_sellers(df)
        topK_pIds = topK_prev_users(model, uId, u_segment, lists_of_bestsellers, df, df_outliers)

        return topK_pIds, pId_int


def eval(df):
    """
    Evaluates the different recommendation algorithms, using RMSE and/or MAPK as chosen in 'options.py'.

    By default (option2=None), a grid of parameters for the ALS Spark algorithm is evaluated. Please edit the grid
    parameters below. The results are written in an output file (see print_file_evaluation_results() in functions.py).

    If option2 (an argument of top5recom.py) is 'baseline' or 'only_custom_BS', then the results of the evaluation
    are printed in stdout.
    """

    # we start with some preparations common to all evaluations:
    df, pId_int, uIds_outliers, df_outliers, old_u = prepare_df(df)
    df_train, df_test = split_train_test_df(df, option)
    df_out_train, df_out_test = split_train_test_df(df_outliers, option) if remove_outliers else (None, None)
    u_ontraintest, p_onlytest = analyse_dfs(df_train, df_test, df_out_train, df_out_test)
    if include_MAPK or option2:
        u_to_eval = np.random.choice(u_ontraintest, size=min(num_u_toeval, len(u_ontraintest)), replace=False)
        if only_last_year: u_to_eval = np.r_[u_to_eval, np.random.choice(old_u, size=int(0.15*num_u_toeval))]
        ll_upurch = obtain_llist_future_u_purchases(pd.concat([df_test, df_out_test]), u_to_eval, p_onlytest, option)
    else:
        u_to_eval = [] ; ll_upurch = []

    # here, if it corresponds, we evaluate models that do not use the collaborative filtering algorithm
    if option2:
        evaluate_model(None, option2, None, df_train, None, ll_upurch, u_to_eval, None, option2=option2, coll_filt=False)
        soft_exit("Model \'%s\' (no collaborative filtering) evaluated." % option2)

    # We need specific files for mllib, we will create them only if not present: (train will be created by train_recommender)
    test_csv_file = get_mlib_fnames(option)[1]
    if not os.path.isfile(test_csv_file):
        if verb: print("Creating test csv file: %s..." % test_csv_file)
        create_mllib_csv(df_test, test_csv_file)
        if verb: print("...file created")
    test = sc.textFile(test_csv_file)
    test = test.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    print("\nStarting grid evaluation...") ; t0grid = time()
    # Parameters for the grid:
    ranks = [25]
    numIterations = [30]
    lamdas = [0.01] # must be a float
    alphas = [60.]  # must be a float

    for rank, numIterations, lamda, alpha in itertools.product(ranks,numIterations,lamdas,alphas):
        print("\n** rank=%i, numIterations=%i, lamda=%.1e, alpha=%.1e\n" % (rank, numIterations, lamda, alpha))
        model, model_name = train_recommender(df_train, option, rank=rank, numIterations=numIterations,
                                              lamda=lamda, alpha=alpha)
        evaluate_model(model, model_name, test, df_train, df_out_train, ll_upurch, u_to_eval, uIds_outliers)

    print("\n...grid evaluated in %.0fs" % (time()-t0grid))


def usage():
    print '''

    DESCRIPTION:

        "top5recom" is a recommender system based on purchase history.

        It recommends products not previously purchased by the consumer, using a combination of collaborative
        filtering and a best-sellers selection based on user segmentation.

        It takes as input a userId and the purchase history in "purchases.csv", and outputs the productIds of
        the 5 most recommended products.


    USAGE:

        To recommend 5 product Ids to a user:

                top5recom.py [userId]               : our recommendation
                top5recom.py [userId] baseline      : baseline recommendation (the 5 all-time best-sellers)
                top5recom.py [userId] only_custom_BS: recommendation using only user-customized best-sellers

                Examples: top5recom.py 374592
                          top5recom.py 374592 baseline

        To evaluate the recommender:

                top5recom.py [option]               : to evaluate our recommender
                top5recom.py [option] baseline      : to evaluate baseline recommendations
                top5recom.py [option] only_custom_BS: to evaluate recommendations based on uId-customized best-sellers

                Examples: top5recom.py eval
                          top5recom.py shorteval baseline

        Troubleshooting:
                - the script requires findspark (among other libraries). If not installed, run from spark instead:
                  $SPARK_HOME/bin/spark-submit top5recom.py [userId]
                - if you got the java error "Too many open files", try "ulimit -n 4096" before running the script.


    OPTIONS:

        When evaluating the recommender, please choose between:
            [option]='eval'         : the full purchase history is used for training and testing
            [option]='shorteval'    : only uses 2014 to speed up the computation
                                     (and only evaluates on that, NOT recommended)

        Options to be defined in the module 'options.py':
            verb                    : (False) set it True do see a short description of the code steps when running
            purch_file              : please set here the path to the purchases.csv file (default: running directory)
            only_last_year          : (True) only trains on 2014 data (like shorteval) but evaluates on all users, treating
                                      the pre-2014 ones as new users and recommending them only best-sellers
            remove_outliers         : (True) removes outliers (7 std from the mean num purchases), see prepare_df()
            include_RMSE            : (False) calculates the root-mean-square error metric
            include_MAPK            : (True) calculates the mean average precision of the top 5 recommendations
            num_u_toeval:           : when calculating MAPK evaluation metrics, we use a reduced number of users
                                      to speed up the computation (default is 1000)


    '''

if __name__ == '__main__':

    allowed_options = ['eval', 'shorteval']
    allowed_options2 = ['baseline', 'only_custom_BS']

    userId=None
    option = None

    try:
        userId = int(sys.argv[1])
    except ValueError:
        # sys.argv[1] is not an integer
        option = sys.argv[1]
        if option not in allowed_options:
            usage()
            soft_exit('\n[ERROR] *** Unknown option \'%s\'. Allowed: %s' % (option,str(allowed_options)), 1)
    except IndexError:
        # sys.argv[1] does not exist
        usage()
        soft_exit('\n[ERROR] *** Please specify an integer userId or an allowed option', 1)

    try:
        option2 = sys.argv[2]
        if option2 not in allowed_options2:
            usage()
            soft_exit('\n[ERROR] *** Unknown option2 \'%s\'. Allowed: %s' % (option2, str(allowed_options2)), 1)
    except IndexError:
        option2 = None

    main(userId, option, option2)

    sc.stop()
