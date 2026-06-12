#ifndef LR_MODEL_H
#define LR_MODEL_H
#include <vector>

using std::vector; 

class LogisticRegression {
public:
    LogisticRegression(
        int num_iter = 1000,
        double alpha = 0.01,
        double tol = 1e-8,
        double threshold = 0.5,
        double lambda_reg = 0.0
    );

    vector<double>& fit(
        vector<vector<double>>& X,
        vector<double>& y
    );

    vector<double> predict_proba(vector<vector<double>>& X);

    vector<int> predict(vector<vector<double>>& X);

    vector<double>& get_theta() { 
        return theta_; 
    }

private:
    double sigmoid(double z);

    vector<double> calculate_gradient(
        vector<double>& theta,
        vector<vector<double>>& X_b,
        vector<double>& y
    );

    double dot_product(
        vector<double>& a,
        vector<double>& b
    );

    vector<double> matrix_vector_mult(
        vector<vector<double>>& mat,
        vector<double>& vec
    );

    vector<double> transpose_vector_mult(
        vector<vector<double>>& mat,
        vector<double>& vec
    );

    vector<double> theta_;
    int num_iter_;
    double alpha_;
    double tol_;
    double threshold_;
    double lambda_reg_;
};

#endif
