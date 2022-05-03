def CPiecewiseQdr(x0,x):
    p_1 = np.percentile(x0,33)
    p_2 = np.percentile(x0,67)
    bins1 = [0, p_1, p_2, 100]
    df_cut, bins = pd.cut(x, bins1, retbins=True, right=False)
    stepsdum = pd.get_dummies(df_cut)
    x_x = pd.concat([x,x], axis = 1)
    x_x_x = np.array(pd.concat([x,x_x], axis = 1))
    x_step_dummies = np.multiply(np.array(stepsdum), x_x_x)
    xstp_2 = x_step_dummies[:,1]
    xstp_2 = xstp_2[xstp_2 > 0]
    xstp_3 = x_step_dummies[:,2]
    xstp_3 = xstp_3[xstp_3 > 0]
    c1 = xstp_2.min()
    c2 = xstp_3.min()
    stepsdum = stepsdum.drop(stepsdum.columns[0], axis = 1)
    stepsdum = sm.add_constant(stepsdum)
    dummies = np.array(stepsdum)
    dummies[:,1] = np.add(dummies[:,2],dummies[:,1])
    dummies0 = dummies
    dummies = np.insert(dummies, 0 ,values = dummies0[:, 0], axis =1)
    dummies = np.insert(dummies, 2 ,values = dummies0[:, 1], axis =1)
    dummies = np.insert(dummies, 4 ,values = dummies0[:, 2], axis =1)
    polynomial_features = PolynomialFeatures(degree=2)
    xp4 = polynomial_features.fit_transform(np.array(x).reshape(-1,1))
    xp4 = xp4[:,1:]
    x_c1 = polynomial_features.fit_transform((np.array(x)-c1).reshape(-1,1))
    x_c1 = x_c1[:,1:]
    x_c2 = polynomial_features.fit_transform((np.array(x)-c2).reshape(-1,1))
    x_c2 = x_c2[:,1:]
    xp4_1 = np.concatenate((xp4,x_c1), axis = 1)
    xp4_2 = np.concatenate((xp4_1,x_c2), axis = 1)
    X_continuous_piecewise_quadratic = np.multiply(dummies, xp4_2)
    X_continuous_piecewise_quadratic = sm.add_constant(X_continuous_piecewise_quadratic)
    return X_continuous_piecewise_quadratic