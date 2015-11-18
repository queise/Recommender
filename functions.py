from options import *

def load_purchase_history():
    """
    Reads file of purchases (path to file set in options.py)
    and returns a dataframe with specific column names.
    """
    df = pd.read_csv(purch_file, sep=";", header=None,
                     names=["timestamp", "userId", "productId"])
    return df


def create_mllib_csv(df, file_name):
    """
    Creates csv file with the format needed by spark mllib
    """
    df.to_csv(file_name, header=None, index=False, columns=["userId", "productId", "purchase"])


def prepare_df(df, special=None):
    """
    Takes the dataframe created by load_purchase_history(), and transforms it by:
     - converting productIds from strings to integers
     - introducing a new column with the "rating" (purchase=1)
     - a new column with the purchase date inferred from the timestamp

     If the argument special=='baseline' it just return the transform dataframe and pId_int (see below).
     Otherwise:
     - removes old data or outliers depending on the 'only_last_year' and 'remove_outliers' in options.py

     Returns
     -------
     df:  the transformed dataframe
     pId_int: a dictionary with the string<->integer equivalences of product Ids
     uIds_outliers: a list with the user Ids of the outliers
     df_outliers: a dataframe with only the outliers
     old_u: a list of the users that were removed if 'only_last_year'=True
    """


    # convert productId strings to integers for mllib algorithm, store equivalences in dict:
    df["productId"] = df["productId"].astype("category")
    pId_int = {}
    for i,pId in enumerate(df.productId.cat.categories, 1):
        pId_int[i] = pId
    df.productId.cat.categories = [i for i in xrange(1,len(df.productId.cat.categories)+1)]

    # introduce a "rating" column because mllib needs it:
    df["purchase"] = 1

    # add a datetime column from the timestamp:
    df['datetime'] = pd.to_datetime(df["timestamp"], unit='s')

    if special=='baseline':
        return df, pId_int

    if only_last_year:
        if verb: print("Based on previous evaluations, we will only consider data from last year.\n")
        date_1yr_before = df.datetime.iloc[-1] + pd.tseries.offsets.DateOffset(years=-1)
        old_u = df.ix[df.datetime<date_1yr_before,:].userId.unique()
        df = df.ix[df.datetime>=date_1yr_before,:]
        old_u = np.setdiff1d(old_u, df.userId.unique())
    else:
        old_u = None

    uIds_outliers = []
    if remove_outliers:
        if verb: print("Removing outliers...")
        uId_and_n_purch = pd.value_counts(df["userId"])
        # two possible methods, they will remove users with more num of purchases than:
        # 1) the mean num purchases + 7 std -> 0.07% out, 5std -> 0.1% out, 3std -> 0.3% out, 2std -> 0.7%
        max_n_purch_per_user = uId_and_n_purch.values.mean() + 7*uId_and_n_purch.values.std()
        #2) a number defined from the quartiles: if *5. -> 3% out (2 NEXT LINES COMMENTED BY DEFAULT)
        #q75, q25 = np.percentile(uId_and_n_purch.values, [75 ,25])
        #max_n_purch_per_user = q75+((q75 - q25)*5.)
        uIds_outliers = uId_and_n_purch.keys()[uId_and_n_purch.values>max_n_purch_per_user].tolist()
        if verb: print("... users with n_purch > %i have been removed. They were a %.2f%% of the users.\n" %
                       (int(max_n_purch_per_user), float(len(uIds_outliers))/len(uId_and_n_purch.values)*100.))
        df_outliers = df.ix[df.userId.isin(uIds_outliers),:]
        df = df.ix[~df.userId.isin(uIds_outliers),:]
    else:
        df_outliers = None

    return df, pId_int, uIds_outliers, df_outliers, old_u


def split_train_test_df(df, option):
    """
    Splits the dataframe in two by date. The sizes depend on the chosen option.

    Returns
    -------
    the train and test dataframes
    """

    if option=='eval':
        # We split chronologically all our data into train and test sets (test accounts for ~20% of the data):
        df_train = df[df.datetime<"2014-05-15"]
        df_test = df[(df.datetime>="2014-05-15")]

    elif option=='shorteval':
        # To gain speed, we only use 2014 data (test also accounts for ~20%)
        df_train = df[(df.datetime>="2014-01-01") & (df.datetime<"2014-06-05")]
        df_test = df[(df.datetime>="2014-06-01")]

    else:
        soft_exit("Option \'%s\' not recognized in function split_train_test_df" % option)

    return df_train, df_test


def analyse_dfs(df_train, df_test, df_out_train, df_out_test):
    """
    Calculates and returns two lists from the dataframes considered for the model (df_train, df_test),
    and the dataframes of users removed as outliers (df_out_train, df_out_test, IF ANY).

    Returns:
    --------
    u_ontraintest: list of users on both train and test sets (including outliers if previously removed)
    p_onlytest: list of products that only appear on the test set

    These lists are important to evaluate the model. By definition the model can only be evaluated on
    users on u_ontraintest, and cannot predict products that the model has not seen on the train set.
    """

    u_ontrain = df_train.userId.unique()
    u_ontest = df_test.userId.unique()
    u_ontraintest = np.intersect1d(u_ontrain, u_ontest)
    if remove_outliers:
        u_ontrain_out = df_out_train.userId.unique()
        u_ontest_out = df_out_test.userId.unique()
        u_ontraintest_out = np.intersect1d(u_ontrain_out, u_ontest_out)
        u_ontraintest = np.unique(np.r_[u_ontraintest, u_ontraintest_out])

    p_ontrain = df_train.productId.unique()
    p_ontest = df_test.productId.unique()
    p_onlytest = np.setdiff1d(p_ontest, p_ontrain)

    return u_ontraintest, p_onlytest


def classify_user(uId, df, uIds_outliers):
    """
    With the help of classify_prev_user(), it classifies the user on 3 segments depending on his purchase activity:
        - "new_user": the user is not on the train dataset
        - "active": users that purchased at least 3 items in the last 3 months
        - "inactive": the rest of users
    If the user is not new, calculates his previous purchases to classify it.

    Returns
    -------
    u_segment: string
    """
    if verb: print("Classifying user in segment by purchase activity...")

    u_all = get_list_users(df, only_all_u=True)

    if (uId not in u_all) and (uId not in uIds_outliers):
        # the user never bought in our store
        u_segment = 'new_user'
        if verb: print("... user classified in segment \'%s\'" % u_segment)
        return u_segment
    else:
        # we get the list of active users:
        u_3p_last3m = get_list_users(df, only_active_u=True)
        return classify_prev_user(uId, u_3p_last3m, uIds_outliers, u_all)


def classify_prev_user(uId, u_3p_last3m, uIds_outliers, u_all, silent=False):
    """
    Classifies users in 'active', 'inactive'. 'outlier' or 'new_user'.

    Parameters
    ----------
    u_3p_last3m: list of all the users that bought at least 3 products in the last 3 months
    uIds_outliers: user Ids of the outliers (removed from training)
    u_all: all user Ids in the train set
    silent: if True, no print out is done even that verb=True. That is to prevent excessive printings when evaluating
            many users at the same time.

    Returns
    -------
    u_segment: string
    """
    if uIds_outliers and (uId in uIds_outliers):
        u_segment = 'outlier'
    elif uId not in u_all:      # only useful when evaluating with only_last_year
        u_segment = 'new_user'
    elif uId in u_3p_last3m:
        u_segment = 'active'
    else:
        u_segment = 'inactive'
    if verb and not silent: print("... user %i classified in segment \'%s\'\n" % (uId, u_segment))

    return u_segment


def topK_prev_users(model, uId, u_segment, lists_of_bestsellers, df_train, df_out_train, coll_filt=True, silent=True):
    """
    Calculates the top5 product recommendations only for users that are already known to the store.
    It combines predictions from a collaborative filtering algorithm with best-sellers from user segmentation.
    The recommended products are always items never purchased by the user.

    Parameters
    ----------
    model: collaborative filtering ALS spark-mllib model, already trained
    uId: user Id, integer
    u_segment: string, either 'active' or 'inactive'
    lists_of_bestsellers: tuple of 3 lists with all-time, last 3 months and last week best-sellers (see calc_ntopK_best_sellers)
    df: dataframe from purchase history (see function load_purchase_history())
    df_out: the outliers

    Returns
    -------
    topK_pIds: product Ids of the top5 recommendations
    """
    # previous purchases of this user:
    if u_segment=='outlier' and coll_filt:
        prev_p = df_out_train[df_out_train.userId==uId].productId.values
    else:
        prev_p = df_train[df_train.userId==uId].productId.values

    # select 5 personalized products from best-sellers lists (last week, last 3 months, all-time) not previously bought:
    bs_pIds = personalized_best_sellers(u_segment, prev_p, lists_of_bestsellers)
    if verb and not silent: print("   %i product recommendations from segment-specific custom selection on best-sellers: %s" %
                                  (len(bs_pIds), str(bs_pIds)))
    # In some cases we do not want recommendations from collaborative filtering. That happens when option2='only_custom_BS'
    # or when the user is an outlier (in both cases, u was flagged 'outlier'). The model was not trained for outliers, so
    # we cannot predict on them. Lets just return recommendations based on best sellers:
    if u_segment=='outlier' or u_segment=='new_user' or not coll_filt:
        return bs_pIds

    # now we complement the best-sellers with recommendations from collaborative filtering:
    num_collfilt = 2    # number of recommendationa from the collaborative filtering algorithm
    tempK = num_collfilt + 2 # most users have purchased only 1 or 2 products, which may be removed if coincident with recommendations
    topK = model.recommendProducts(uId, tempK)
    topK_pIds = [x[1] for x in topK]
    # removes products previously purchased by this user and those already recommended by best-sellers:
    topK_pIds = [pId for pId in topK_pIds if (pId not in prev_p) and (pId not in bs_pIds)]
    if not isinstance(topK_pIds, list): topK_pIds = [ topK_pIds ] # avoids future error in case the list is empty
    # we will keep adding the number of recommended products until we find num_collfilt(=2) not already purchased:
    while len(topK_pIds) < num_collfilt:
        tempK = tempK*2
        topK = model.recommendProducts(uId, tempK)
        topK_pIds = [x[1] for x in topK]
        topK_pIds = [pId for pId in topK_pIds if (pId not in prev_p) and (pId not in bs_pIds)]
        if not isinstance(topK_pIds, list): topK_pIds = [ topK_pIds ]
    topK_pIds = topK_pIds[:num_collfilt]
    if verb and not silent: print("   %i product recommendations from collaborative filtering: %s\n" % (len(topK_pIds), str(topK_pIds)))

    topK_pIds.extend(bs_pIds[:(5-num_collfilt)])

    return topK_pIds


def topK_no_coll_filt(df, df_outliers, uId, u_segment):
    """
    Calculates the top 5 product recommendations for users segmented as outliers.
    As the outliers are removed from the trainning of the model, the model cannot predict on them, so instead
    we just make a selection from best-sellers, removing previous purchases of the user..

    Returns
    -------
    top5_alltime: a list of the 5 products Ids
    """

    lists_of_bestsellers = calc_ntopK_best_sellers(df)
    # previous purchases of this user:
    if u_segment=='outlier':
        prev_p = df_outliers[df_outliers.userId==uId].productId.values
    else:
        prev_p = df[df.userId==uId].productId.values

    return personalized_best_sellers(u_segment, prev_p, lists_of_bestsellers)


def personalized_best_sellers(u_segment, prev_p, lists_of_bestsellers):
    """
    Calculates 5 best-sellers with criteria depending on the user segmentation.
    The products are always items never purchased by the user.

    Parameters
    ----------
    u_segment: string, either 'active', 'inactive', 'outlier' or 'baseline'
    prev_p: product Ids of previous purchases of the user
    lists_of_bestsellers: tuple of 3 lists with all-time, last 3 months and last week best-sellers
                          (see calc_ntopK_best_sellers)

    Returns
    -------
    bs_pIds: the product Ids of the selected best-sellers

    """
    # really they are not top5 but top25 (to assure we have enough items after removing previous purchases)
    top5_alltime, top5_3months, top5_1week = lists_of_bestsellers

    bs_pIds = []

    if u_segment=='baseline':
        while len(bs_pIds) < 5:
            bs_pIds.append(next((pId for pId in top5_alltime if ((pId not in prev_p) and (pId not in bs_pIds))), 'ErrorTop5'))

    elif u_segment=='active' or u_segment=='outlier':
        bs_pIds.append(next((pId for pId in top5_alltime if ((pId not in prev_p) and (pId not in bs_pIds))), 'ErrorTop5'))
        bs_pIds.append(next((pId for pId in top5_3months if (pId not in prev_p) and (pId not in bs_pIds)), 'ErrorTop5'))
        bs_pIds.append(next((pId for pId in top5_1week if (pId not in prev_p) and (pId not in bs_pIds)), 'ErrorTop5'))
        bs_pIds.append(next((pId for pId in top5_3months if (pId not in prev_p) and (pId not in bs_pIds)), 'ErrorTop5'))
        bs_pIds.append(next((pId for pId in top5_3months if (pId not in prev_p) and (pId not in bs_pIds)), 'ErrorTop5'))

    elif u_segment=='inactive' or u_segment=='new_user':
        bs_pIds.append(next((pId for pId in top5_alltime if ((pId not in prev_p) and (pId not in bs_pIds))), 'ErrorTop5'))
        bs_pIds.append(next((pId for pId in top5_3months if ((pId not in prev_p) and (pId not in bs_pIds))), 'ErrorTop5'))
        bs_pIds.append(next((pId for pId in top5_3months if ((pId not in prev_p) and (pId not in bs_pIds))), 'ErrorTop5'))
        bs_pIds.append(next((pId for pId in top5_3months if ((pId not in prev_p) and (pId not in bs_pIds))), 'ErrorTop5'))
        bs_pIds.append(next((pId for pId in top5_3months if ((pId not in prev_p) and (pId not in bs_pIds))), 'ErrorTop5'))

    else:
        soft_exit("ERROR: User segment \'%s\' not understood in personalized_best_sellers()" % u_segment)

    assert len(bs_pIds)==5

    return bs_pIds


def calc_ntopK_best_sellers(df, which='all', ntopK=25):

    # The all-time top 25 purchased products:
    top5_alltime =  pd.value_counts(df["productId"])[:ntopK].keys().tolist()
    if which=='alltime': return top5_alltime
    # The top 25 purchased products in the last 3 months:
    top5_3months =  pd.value_counts(df.ix[df.datetime>"2014-03-25","productId"])[:ntopK].keys().tolist()
    # The top 25 purchased products in the last week are:
    top5_1week = pd.value_counts(df.ix[df.datetime>"2014-06-18","productId"])[:ntopK].keys().tolist()

    return top5_alltime, top5_3months, top5_1week


def train_recommender(df, option, rank=25, numIterations=30, lamda=0.01, alpha=60.):
    """
    Trains a collaborative filtering recommender using the ALS.trainImplicit algorithm of the spark mllib library. This
    implementation of the alternating least squares (ALS) algorithm uses implicit ratings that are related to the level
    of confidence in observed user preferences, rather than explicit ratings given to items.

    Parameters of the ALS.trainImplicit recommender
    -----------------------------------------------
    rank: number of latent factors in the model
    iterations: number of iterations of ALS (recommended: 10-20)
    lambda: regularization factor (recommended: 0.01)
    alpha: confidence parameter on the given ratings (a float)
    blocks: number of blocks used to parallelize the computation (set to -1 to auto-configure, but set to 1 to be safe)

    Returns
    -------
    model: the model, trained
    model_name: string
    """

    model_name = "ALS_r%i_i%i_l%.1e_a%.1e" % (rank, numIterations, lamda, alpha)
    if option: model_name += '_' + option
    if remove_outliers: model_name += '_out'
    if only_last_year: model_name += '_1yr'
    if verb: print("Collaborative filtering model name: %s\n" % model_name)

    # train model only if it has not yet been trained:
    if not os.path.exists(model_name):

        # we need a specific file for mllib, create it only if not present:
        train_csv_file = get_mlib_fnames(option)[0]
        if not os.path.isfile(train_csv_file):
            if verb: print("Creating train_csv_file: %s..." % train_csv_file)
            create_mllib_csv(df, train_csv_file)
            if verb: print("...file created")

        # create and train model:
        if verb: print("Training model...") ; t0 = time()
        train = sc.textFile(train_csv_file)
        train = train.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
        model = ALS.trainImplicit(train, rank, iterations=numIterations, lambda_=lamda, blocks=1, alpha=alpha, seed=myseed)
        if verb: print('...model trained in %.1fs\n' % (time()-t0))

        # save model
        model.save(sc, model_name)

    # if already trained, just load it:
    else:
        if verb: print('Loading previously trained model...')
        model = MatrixFactorizationModel.load(sc, model_name)
        if verb: print('...model loaded\n')

    return model, model_name


def get_list_users(df, only_all_u = False, only_active_u=False):
    """
    Creates a list with the users in df that bought at least 3 products in the last 3 months.
    It also return a list of all the users in the df if required.
    """

    if not only_active_u:
        # First, a list with all the users:
        all_u = list(df.userId.unique())
        if only_all_u:
            return all_u

    # Now, we create a new dataframe with only the last purchase of each user:
    df_lastp = df.ix[:,1:].sort_values('datetime').drop_duplicates(subset=['userId'],keep='last')

    # to find the active users, drop those that did not bought during the last 3 months.
    date_3m_bef_last_purch = df.datetime.iloc[-1] + pd.tseries.offsets.DateOffset(months=-3) # df.datetime.dt.date.max()
    u_1p_last3m = df_lastp.ix[df.datetime>date_3m_bef_last_purch,:].userId.tolist()
    df_u3m = df.ix[df.userId.isin(u_1p_last3m),1:]
    # we remove the last purchase of each user from df_u3m, two times:
    for i in xrange(2):
        df2rem = df_u3m.sort_values('datetime').drop_duplicates(subset=['userId'],keep='last')
        df2rem['key'] = 'x'
        df_u3m = pd.merge(df_u3m, df2rem, on=df_u3m.columns.tolist(), how='left')
        df_u3m = df_u3m[df_u3m['key'].isnull()].drop('key', axis=1)
    # if the last purchase now is still within the last three months, then the user bought 3 items in the last 3 months:
    df_u3m = df_u3m.sort_values('datetime').drop_duplicates(subset=['userId'],keep='last')
    u_3p_last3m = df_u3m.ix[df_u3m.datetime>date_3m_bef_last_purch,:].userId.tolist()

    if only_active_u:
        return u_3p_last3m
    else:
        return u_3p_last3m, all_u


def obtain_llist_future_u_purchases(df_test, u_to_eval, p_onlytest, option):
    """
    Infers the future purchases of a list of users, writes it in a file and returns it.
    If that list was calculated before, it just reads it from a file.

    Parameters
    ----------
    df_test: the dataframe with the future purchases, not seen by the training
    u_to_eval: the users for which it calculates the future purchases
    p_onlytest: products only on test set are removed for the evaluation, because they cannot be predicted.

    Returns
    -------
    ll_upurch: a list of sublists, each sublist being the future purchases of each user in u_to_eval
    """
    ll_upurch = []
    ll_fname = "ll_ufpurch_" + option + str(len(u_to_eval)) + '.csv'
    if remove_outliers: ll_fname = ll_fname[:-4] + '_out.csv'
    if only_last_year: ll_fname = ll_fname[:-4] + '_1yr.csv'

    if os.path.isfile(ll_fname):
        if verb: print("Reading ll_upurch from file %s..." % ll_fname) ; sys.stdout.flush()
        with open(ll_fname, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                ll_upurch.append([int(elem) for elem in row])
        return ll_upurch

    else:
        if verb: print("Creating ll_upurch...") ; sys.stdout.flush()
        for uId in u_to_eval: # 30 mins for all users
            futu_purch_pIds = np.unique(df_test[df_test.userId==uId].productId.values)
            ll_upurch.append([pId for pId in futu_purch_pIds if pId not in p_onlytest])

        if verb: print("Storing ll_upurch to file %s..." % ll_fname); sys.stdout.flush()
        with open(ll_fname, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(ll_upurch)
        return ll_upurch


def evaluate_model(model, model_name, test, df_train, df_out_train, ll_upurch, u_to_eval, uIds_outliers,
                   option2=None, coll_filt=True):
    """
    Evaluation of the model using the metrics RMSE and/or MAPK.
    The results are printed in an output file (see print_file_evaluation_results()).

    Some parameters for MAPK
    ------------------------
    uIds_outliers: outliers will get best-seller evaluations instead of model predictions (were removed from train)
    u_to_eval: the users for which it evaluates the model
    ll_upurch: a list of sublists, each sublist being the future purchases of each user in u_to_eval

    """

    t0e = time() ; RMSE = None ; MAPK = None
    if include_RMSE and coll_filt:
        # Evaluation: RMSE on all test predictions
        if verb: print("Predicting rating for all test entries and RMSE...") ; t0 = time()
        X_test = test.map(lambda r: (r[0], r[1]))
        y_pred = model.predictAll(X_test).map(lambda r: ((r[0], r[1]), r[2]))
        test_and_pred = test.map(lambda r: ((r[0], r[1]), r[2])).join(y_pred)
        #MSE = test_and_pred.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        #if verb: print("%s MSE = %.5f" % (model_name, MSE))
        #RMSE = m.sqrt(MSE) # RMSE is in the same units as the purchase 0-1
        RMSE = calc_RMSE(test_and_pred)
        print("%s RMSE = %.5f in %.0fs" % (model_name, RMSE, (time()-t0e)))
        del X_test, y_pred, test_and_pred

    if include_MAPK or not coll_filt:
        if verb: print("Predicting top 5 recommendations for a sample of %i users..." % len(u_to_eval)) ; t0 = time()
        lists_of_bestsellers = calc_ntopK_best_sellers(df_train)
        u_3p_last3m, all_u = get_list_users(df_train)
        ll_predict = []
        for uId in u_to_eval: # extremely slow when going through all users, choose number in options.py
            if option2=='baseline':
                topK = topK_prev_users(model, uId, 'baseline', lists_of_bestsellers, df_train, df_out_train, coll_filt=coll_filt)
            else:
                u_segment = classify_prev_user(uId, u_3p_last3m, uIds_outliers, all_u, silent=True)
                topK = topK_prev_users(model, uId, u_segment, lists_of_bestsellers, df_train, df_out_train, coll_filt=coll_filt)
            ll_predict.append(topK)
        if verb: print("calculating MAPK...") ; sys.stdout.flush()
        MAPK = mapk(ll_upurch, ll_predict)
        print("%s MAPK = %.5f in %.0fs" % (model_name, MAPK, (time()-t0e)))
        sys.stdout.flush()

    print_file_evaluation_results(model_name, RMSE, MAPK, t0e, coll_filt)
    del model


def print_file_evaluation_results(model_name, RMSE, MAPK, t0e, coll_filt):
    """
    Yes, it prints the files with the results of the evaluation.
    """
    if include_RMSE and include_MAPK and coll_filt:
        with open("ALS_RMSE-MAPK_results.csv", 'a') as f:
            f.write("{:40s} {:6.1f} {:9.4f} {:10.5f}\n".format(model_name, (time() - t0e)/60., RMSE, MAPK))
    elif include_RMSE and coll_filt:
        with open("ALS-RMSE_results.csv", 'a') as f:
            f.write("{:40s} {:6.1f} {:9.4f}\n".format(model_name, (time() - t0e)/60., RMSE))
    elif include_MAPK or not coll_filt:
        with open("ALS-MAPK_results.csv", 'a') as f:
            f.write("{:40s} {:6.1f} {:10.5f}\n".format(model_name, (time() - t0e)/60., MAPK))


def evaluate_baseline_MAPK(ll_upurch, top5_alltime):
    """
    Evaluates the MAPK score for  baseline predictions.
    The baseline predicts always the top5_alltime.
    MAPK compares the predictions with the real future purchases of the users ll_upurch.
    """
    ll_predict =  [top5_alltime for l_upurch in ll_upurch]
    MAPK = mapk(ll_upurch, ll_predict, k=5)
    return MAPK


def calc_RMSE(test_and_pred):
    """
    Calculates RMSE (root-mean-square error)
    Before, it corrects predictions to be between [0,1]

    Parameters
    ----------
    test_and_pred : spark RDD with row format (uid, pid), (y_pred, y_test)
                    only the second tuple is used, y_test are the known ratings
    Returns
    -------
    RMSE
    """
    arr = np.array(test_and_pred.collect())
    # correction for predictions below 0 or above 1:
    arr[:,1,1][arr[:,1,1] > 1] = 1.
    arr[:,1,1][arr[:,1,1] < 0] = 0.
    MSE = np.mean((arr[:,1,0] - arr[:,1,1])**2)
    RMSE = m.sqrt(MSE)
    return RMSE


def get_mlib_fnames(option):
    """
    Return the file names that will be used by the ALS soark mllib algorithm.
    The filenames include the option if any (eval/shorteval), and '_out' if otliers are removed, because both
    things change the users in the files.
    """
    train_csv_file='upp_train.csv'
    test_csv_file='upp_test.csv'

    if option:
        train_csv_file = train_csv_file[:-4] + '_' + option + '.csv'
        test_csv_file  = test_csv_file[:-4] + '_' + option + '.csv'
    if remove_outliers:
        train_csv_file = train_csv_file[:-4] + '_out.csv'
        test_csv_file  = test_csv_file[:-4] + '_out.csv'
    if only_last_year:
        train_csv_file = train_csv_file[:-4] + '_1yr.csv'
        test_csv_file  = test_csv_file[:-4] + '_1yr.csv'

    return train_csv_file, test_csv_file


def print_topK_pIds(topK_pIds, pId_int):
    """
    Prints the topK_pIds, after converting them from the integer form to the original string format.

    Parameters:
        - topK_pIds: list of K product recommendations, where productId was codified to integer
        - pId_int: dictionary, keys are the pId integers, values are their string pId equivalents
    """
    if verb: print("The top 5 product recommendations are:")
    for int_pId in topK_pIds:
        print pId_int[int_pId]


def apk(actual, predicted, k=5):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    from Ben Hamner github.com/benhamner/Metrics/tree/master/Python/ml_metrics
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    from Ben Hamner github.com/benhamner/Metrics/tree/master/Python/ml_metrics
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def soft_exit(exit_message, exit_code=None):
    """
    Exits the program printing an error message
    and stopping SparkContext.
    """
    print(exit_message)
    print("Exiting quietly...")
    sc.stop()
    sys.exit(exit_code)
