#include "lr_model.h"

using namespace std; 

LogisticRegression::LogisticRegression(
    int num_iter, 
    double alpha, 
    double tol,
    double threshold, 
    double 
    lambda_reg
): num_iter_(num_iter), alpha_(alpha), tol_(tol), threshold_(threshold), lambda_reg_(lambda_reg) {}

double LogisticRegression::sigmoid(double z) {
    if (z < -500) z = 500;
    else if (z > 500) z = 500;
    return 1.0 / (1.0 + exp(-z));
}

double LogisticRegression::dot_product(
    vector<double>& a,vector<double>& b
) {
    double result = 0.0;
    for (int i = 0; i < a.size(); ++i)
        result += a[i] * b[i];
    return result;
}

vector<double> LogisticRegression::matrix_vector_mult(
    vector<vector<double>>& mat,
    vector<double>& vec
) {

    vector<double> result(mat.size(), 0.0);
    for (int i = 0; i < mat.size(); ++i)
        result[i] = dot_product(mat[i], vec);
    return result;
}

vector<double> LogisticRegression::transpose_vector_mult(
    vector<vector<double>>& mat,
    vector<double>& vec
) {

    if (mat.empty()) return {};
    int ncols = mat[0].size();
    vector<double> result(ncols, 0.0);
    for (int i = 0; i < mat.size(); ++i)
        for (int j = 0; j < ncols; ++j)
            result[j] += mat[i][j] * vec[i];
    return result;
}

vector<double> LogisticRegression::calculate_gradient(
    vector<double>& theta,
    vector<vector<double>>& X_b,
    vector<double>& y
) {

    int m = y.size();
    vector<double> h = matrix_vector_mult(X_b, theta);
    for (int i = 0; i < m; ++i)
        h[i] = sigmoid(h[i]) - y[i];

    vector<double> grad = transpose_vector_mult(X_b, h);
    for (int j = 0; j < grad.size(); ++j)
        grad[j] /= (double)(m);

    for (int j = 1; j < grad.size(); ++j)
        grad[j] += (lambda_reg_ / (double)(m)) * theta[j];

    return grad;
}

vector<double>& LogisticRegression::fit(
    vector<vector<double>>& X,
    vector<double>& y
) {

    int m = X.size();
    int n = X.empty() ? 0 : X[0].size();

    vector<vector<double>> X_b(m, vector<double>(n + 1, 1.0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            X_b[i][j + 1] = X[i][j];

    theta_.assign(n + 1, 0.0);

    for (int iter = 0; iter < num_iter_; ++iter) {
        auto grad = calculate_gradient(theta_, X_b, y);

        for (int j = 0; j < theta_.size(); ++j)
            theta_[j] -= alpha_ * grad[j];

        double norm = 0.0;
        for (double g : grad)
            norm += g * g;
        if (sqrt(norm) <= tol_)
            break;
    }

    return theta_;
}

vector<double> LogisticRegression::predict_proba(vector<vector<double>>& X) {

    int m = X.size();
    int n = X.empty() ? 0 : X[0].size();

    vector<vector<double>> X_b(m, vector<double>(n + 1, 1.0));
        for (int i = 0; i < m; ++i)
    
        for (int j = 0; j < n; ++j)
            X_b[i][j + 1] = X[i][j];

    vector<double> probs = matrix_vector_mult(X_b, theta_);
    for (int i = 0; i < m; ++i)
        probs[i] = sigmoid(probs[i]);
    return probs;
}

vector<int> LogisticRegression::predict(vector<vector<double>>& X) {

    auto probability = predict_proba(X);
    vector<int> result(probability.size());
    for (int i = 0; i < probability.size(); ++i)
        result[i] = probability[i] >= threshold_ ? 1 : 0;

    return result;
}

