/*CNN forward and backward
Q. Xu, Dec 25, 2022
Notes:
1) it is a rapid prototype only with low-level computation without HPC, as an illustration of the computation procedure
2) backward portion is being worked on and will be updated ASAP in Jan.
Reference: Andre NG, "Deep Learning Specilization", Coursera
*/

#include <iostream>
#include <vector>
#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include<string.h>
#include <random>

using namespace std;
typedef vector<vector<vector<vector<float>>>> v4f; //4D vector
typedef vector<vector<vector<float>>> v3f; //3D vector
typedef vector<vector<float>> v2f; //3D vector
typedef vector<float> v1f;

class math_operations {
public:
    v2f activation_wrapper (v2f& Window, string mode) {
        int row = Window.size(), col = Window[0].size ();
        v2f A (row, v1f(col,0));
        if (mode == "sigmoid")
            activation_sigmoid (Window, row, col, A);
        else if (mode == "relu")
            activation_relu (Window,row, col, A);
        else if (mode == "softmax")
            activation_softmax (Window, row, col, A);
        else
            return {{-1},{-1}};
        return A;
    }

    float mean_matrix_window (v3f& a_prev_slice, int vert_start, int vert_end, int horiz_start, int horiz_end, int c) {
        float mean_val = 0;
        for (int i = vert_start; i < vert_end; i++) {
            for (int j = horiz_start; j < horiz_end; j++) {
                mean_val += a_prev_slice[i][j][c];
            }
        }
        return mean_val;
    }

    //pad_zero to a 2d array
    void pad_2dArr (deque <deque <float>>& X, int n_pad, float pad_val) {
        int row = X.size(), col = X[0].size();
        /*inputs: n_pad - pad size, X - 2D vector */
        for (int i = 0; i < row; i++) {
            for (int p = 0; p < n_pad; p++) {
                X[i].push_front(pad_val);
                X[i].push_back(pad_val);
            }
        }
        
        //pad on front and end rows
        deque <float> row_pad (col + n_pad, pad_val);
        for (int j = 0; j < n_pad; ++j) {
            X.push_front(row_pad);
            X.push_back(row_pad);
        }
    }

    //matrix element_wise multiply
    float v3f_multiply_element_wise (v3f& M1, v3f& M2) {
        int row = M1.size(), col = M1[0].size(), depth = M1[1].size();
        //v3f M1M2 (row, v2f (col,v1f(depth, 0)));
        float sum_val = 0;
        for (int i = 0; i < row * col * depth; i++) {
            int x = i%(row*col)/row, y= i % (row*col)%row, z = i/(row*col);
            sum_val += M1[x][y][z]* M2[x][y][z];
        }
        return sum_val;
    }

private:
    void activation_sigmoid (v2f& Window, int row, int col, v2f& A) {
        for (int i = 0; i < row*col; i++) {
            int x = i / row, y = i % row;
                A[x][y] = 1 / (1 + exp(-Window[x][y]));
        }
    }

    void activation_relu (v2f& Window, int row, int col, v2f& A) {
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % row;
            float value = Window[x][y];
            A[x][y] = value > 0? value: 0;
        }
    }

    void activation_softmax (v2f& Window, int row, int col, v2f& A) {
        float sum_val = 0;
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % row;
            float value_single = exp (Window[x][y]);
            A[x][y] = value_single;
            sum_val += value_single;
        }
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % row;
            A[x][y] /= sum_val;
            }
    }
};

struct hparameters {    
    int n_pad = 2;
    float pad_val = 0.0;
};

class forward_cnn : public math_operations {
    int n_batch = 0;
public:
    forward_cnn(int batch_size) {
        int n_batch = batch_size;
    }

    //convolution for a single step, Z = W*X + b, * is element-wise multiply of matrix
    float conv_single_step(v3f& Slice_window, v3f& W, float b) {
        //Slice_window: fh x fw x channel, W: weights
        float z = v3f_multiply_element_wise(Slice_window, W);
        return z + b;
    }

    template <typename T>
    pair <v4f, tuple <v4f, v3f, float, unordered_map <string, T> >> conv_forward(v4f& A_prev, v3f* W, float b, unordered_map <string, T>& hparameters) {
        int  m = A_prev.size(), n_H_prev = A_prev[0].size(), n_W_prev = A_prev[1].size(), n_C_prev = A_prev[2].size();
        int fh = W.size(), fw = W[0].size(), n_C_prev = W[1].size(), n_C = W[2].size();
        int stride = hparameters["stride"];
        int n_pad = hparameters["pad"];
        float pad_val = hparameters["pad_val"];
        int n_H = (int)((n_H_prev + 2 * pad - f) / stride + 1);
        int n_W = (int)((n_W_prev + 2 * pad - f) / stride + 1);

        v4f Z(m, v3f(n_H, v2f(n_W, v1f(n_C, 0))));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n_C_prev; j++) {
                v2f X(n_C_prev, v1f(n_C_prev, 0));
                for (int k = 0; k < n_C_prev; k++) {
                    for (int m = 0; m < n_C_prev; m++) {
                        v2f X[j][k] = A_prev[j][k];
                    }
                }
                pad_2dArr(X, n_pad, pad_val);
            }
        }

        for (int i = 0; i < m; i++) {
            v3f a_prev_pad = A_prev_pad[i];
            for (int h = 0; h < n_H; h++) {
                int vert_start = stride * h;
                int vert_end = vert_start + f;
                for (int w = 0; w < n_W; w++) {
                    int horiz_start = stride * w;
                    int horiz_end = horiz_start + f;
                    for (int c = 0; c < n_C; c++) {
                        v3f a_slice_prev = A_prev_pad[i, vert_start:vert_end, horiz_start : horiz_end, : ];
                        v3f weights = W[:, : , : , c];
                        v3f biases = b[:, : , : , c];
                        Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases);
                    }
                }
            }
        }
        cache = make_tuple(A_prev, W, b, hparameters);
        return make_pair(Z, cache);
    }

    template <typename T>
    tuple <v4f, pair<v4f, unordered_map <string, T>> > pool_forward(v4f& A_prev, unordered_map <string, T>& hparameters, string mode, unordered_map <string, v2f>& cache) {
        int m = n_H_prev = n_W_prev = n_C_prev = A_prev.size();
        int f = hparameters["f"];
        int stride = hparameters["stride"];

        int n_H = (int)(1 + (n_H_prev - f) / stride);
        int n_W = (int)(1 + (n_W_prev - f) / stride);
        int n_C = n_C_prev;

        //initilize
        v4f A = (m, v3f(n_H, v2f(n_W, v1f(n_C, 0))));
        for (int i = 0; i < m; i++) {
            for (int h = 0; h < n_H; h++) {
                int vert_start = stride * h;
                int vert_end = vert_start + f;
                for (int w = 0; w < n_W; w++) {
                    horiz_start = stride * w;
                    horiz_end = horiz_start + f;
                    for (int c = 0; c < n_C; c++) {
                        float a_prev_slice = A_prev[i];
                        if (mode == "max") {
                            A[i, h, w, c] = v3f_max(a_prev_slice[vert_start:vert_end, horiz_start : horiz_end, c]);
                        }
                        else if (mode == "average") {
                            A[i, h, w, c] = v3f_mean(a_prev_slice[vert_start:vert_end, horiz_start : horiz_end, c]);
                        }
                    }
                }
            }
        }
        cache = make_pair(A_prev, hparameters);
        return make_tuple(A, cache);
    }
};
