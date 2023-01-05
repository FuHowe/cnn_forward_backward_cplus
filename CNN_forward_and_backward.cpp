/*
CNN forward and backward
Qinwu Xu, Dec 25, 2022

Notes:
1) it is a rapid prototype with low-level computation and parallism for CPUs
Major methods refered to: Andre NG, "Deep Learning Specilization", Coursera

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
#include <omp.h>
#include <limits>

using namespace std;
typedef vector<vector<vector<vector<float>>>> v4f; //4D vector
typedef vector<vector<vector<float>>> v3f; //3D vector
typedef vector<vector<float>> v2f; //2D vector
typedef vector<float> v1f;

class math_operations {
public:
    //a wrapper function to call activation given the mode type
    v2f activation_wrapper (v2f& Window, string mode) {
        int row = Window.size(), col = Window[0].size ();
        v2f A (row, v1f(col,0));
        if (mode == "sigmoid")
            activation_sigmoid(Window, row, col, A);
        else if (mode == "relu")
            activation_relu(Window, row, col, A);
        else if (mode == "softmax")
            activation_softmax(Window, row, col, A);
        else
            cout << "Error: no valid activation type [sigmoid, relu, softmax] is provided" << endl;
            return {{-1},{-1}};
        return A;
    }
    //mean value of a 2D vector given the start and end position of each axis
    float mean_matrix_window (v3f& a_prev_slice, int vert_start, int vert_end, int horiz_start, int horiz_end, int c) {
        float mean_val = 0;
        for (int i = vert_start; i < vert_end; i++) {
            for (int j = horiz_start; j < horiz_end; j++) {
                mean_val += a_prev_slice[i][j][c];
            }
        }
        return mean_val;
    }

    // pad a 4d vector with padding size of n_pad and value of pad_val
    void pad_4d_vector (v4f& X, int n_pad, float pad_val, v4f & X_padded) {
        //X: 4d vector with size of m x n_H x n_w x n_c
        // goal: pad each 2d slice of the 4d vector (n_H x n_W) with pad size of n_pad and value of pad_val
        int m = X.size(), n_H = X[0].size(), n_W = X[1].size(), n_c = X[2].size(); // # of sample, # of channels
        
        #pragma omp parallel for collapse(4)
        for (int i = 0; i < m; i++) {
            for (int h = n_pad; h < n_H + n_pad; h++) {
                for (int w = n_pad; w < n_W + n_pad; w++) {
                    for (int c = 0; c < n_c; c++) {
                        X_padded[i][h][w][c] = X[i][h - n_pad][w - n_pad][c];
                    }
                }
            }
        }
    }

    float get_max_of_2dv(v2f & X) {
        float max_val = numeric_limits <float>::max();
        for (int i = 0; i < X.size(); i++) {
            for (int j = 0; j < X[0].size(); j++) {
                if (X[i][j] > max_val) {
                    max_val = X[i][j];
                }
            }
        }
        return max_val;
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

template <typename T>
class forward_cnn : public math_operations {
    int n_batch = 0;
public:
    forward_cnn(int batch_size) {
        int n_batch = batch_size;
    }

    //convolution for a single step, Z = W*X + b, * is element-wise multiply of matrix
    float conv_single_step(v3f& Slice_window, v3f& W, float b) {
        //Slice_window: fh x fw x channel, W: weights
        float z_val = v3f_multiply_element_wise(Slice_window, W);
        return z_val + b;
    }

    pair <v4f, tuple <v4f, v3f, float, unordered_map <string, T> >> conv_forward(v4f& A_prev, v3f* W, v4f & b, unordered_map <string, T>& hparameters) {
        int  m = A_prev.size(), n_H_prev = A_prev[0].size(), n_W_prev = A_prev[1].size(), n_C_prev = A_prev[2].size();
        int fh = W.size(), fw = W[0].size(), n_C_prev = W[1].size(), n_C = W[2].size();
        int stride = hparameters["stride"];
        int n_pad = hparameters["pad"];
        float pad_val = hparameters["pad_val"];
        int n_H = (int)((n_H_prev + 2 * n_pad - fh) / stride + 1);
        int n_W = (int)((n_W_prev + 2 * n_pad - fw) / stride + 1);

        v4f Z(m, v3f(n_H, v2f(n_W, v1f(n_C, 0))));
        //padding the activation of previous layer
        v4f A_prev_pad (m, v3f(n_H_prev + 2*n_pad, v2f(n_W_prev + 2*n_pad, v1f(n_C_prev, 0))));
        pad_4d_vector(A_prev, n_pad, pad_val, A_prev_pad);

        // apply convolution by sliding filter windows across all batch samples and height/width of each sample
        #pragma omp parallel for
        for (int i = 0; i < m; i++) { //loop over batch sample
            v3f a_prev_pad = A_prev_pad[i]; 
            for (int h = 0; h < n_H; h++) {
                int vert_start = stride * h;
                int vert_end = vert_start + fh;
                for (int w = 0; w < n_W; w++) {
                    int horiz_start = stride * w;
                    int horiz_end = horiz_start + fw;
                    for (int c = 0; c < n_C; c++) {

                        v3f a_slice_prev(vert_end - vert_start, v2f(horiz_end - horiz_start, v1f(n_C_prev, 0)));
                        for (int L=vert_start; L<vert_end; L++)
                            for (int M = horiz_start; M < horiz_end; M++)
                                a_slice_prev [L][M][c] = a_prev_pad [L][M][c];

                        v3f weights(fh, v2f (fw, v1f(n_C, 0)));
                        for (int L = vert_start; L < vert_end; L++)
                            for (int M = horiz_start; M < horiz_end; M++)
                                weights =  W[L][M][c];

                        v3f biases (1, v2f(1, v1f(n_C, 0)));
                        v3f biases = b[i][0][0][c];
                        Z[i][h][w][c] = conv_single_step (a_slice_prev, weights, biases);
                    }
                }
            }

        }
        tuple <v4f, v4f, v4f, unordered_map <string, T> > cache = make_tuple (A_prev, W, b, hparameters);
        return make_pair(Z, cache);
    }

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

template <typename T>
class conv_backward_computation : public math_operations {
public:
    tuple <v4f, v4f, v4f> conv_backward(v4f& dZ, tuple <v4f, v4f, v4f, unordered_map <string, T>> cache) {

    // Retrieve information from "cache"
        v4f A_prev = get <0>(cache), W = get<1>(cache), b = get<2>(cache);
        unordered_map <string, T> hparameters = get<3>(cache);

        // Dimensions activation of previous layer: m: # of batches, H x W: height by height, C: channel #
        int m = A_prev.size(), n_H_prev = A_prev[0].size(), n_W_prev = A_prev[1].size(), n_C_prev = A_prev[2].size();
        // Dimension of filter at current layer: fh by fw
        int fh = W.size(), fw = W[0].size(), n_C_prev = W[1].size(), n_C = W[2].size();
        int stride = hparameters["stride"];
        int n_pad = hparameters["pad"];
        float pad_val = hparameters["pad_val"];
        int m = dZ.size(), n_H = dZ[0].size(), n_W = dZ[1].size(), n_C = dZ[2].size();

        v4f dA_prev(m, v3f(n_H_prev, v2f(n_W_prev, v1f(n_C_prev, 0)))); //gradient of cost wrt previous layer activation
        v4f dW(fh, v3f(fw, v2f(n_C_prev, v1f(n_C, 0)))); //gradient of cost wrt W
        v4f db(1, v3f(1, v2f(1, v1f(n_C, 0)))); //

        v4f A_prev_pad(m, v3f(n_H_prev + 2 * n_pad, v2f(n_W_prev + 2 * n_pad, v1f(n_C_prev, pad_val))));
        pad_4d_vector(A_prev, n_pad, pad_val, A_prev_pad);
        v4f dA_prev_pad(m, v3f(n_H_prev + 2 * n_pad, v2f(n_W_prev + 2 * n_pad, v1f(n_C_prev, pad_val))));
        pad_4d_vector(dA_prev, n_pad, pad_val, dA_prev_pad);

        for (int i = 0; i < m; i++) { // # loop over the training examples
            v3f a_prev_pad = A_prev_pad[i];
            v3f da_prev_pad = dA_prev_pad[i];

            for (int h = 0; h < n_H; h++) { //img height
                for (int w = 0; w < n_W; w++) { //img width
                    for (int c = 0; c < n_C; c++) { //channels
                        int vert_start = stride * h;
                        int vert_end = vert_start + fh;
                        int horiz_start = stride * w;
                        int horiz_end = horiz_start + fw;

                        v3f a_slice(vert_end - vert_start, v2f(horiz_end - horiz_start, v1f(n_C, 0)));
                        a_slice = { a_prev_pad.begin() + vert_start, a_prev_pad.begin() + vert_end };
                        for (int M = vert_start; M < vert_end; M++) {
                            a_slice[M] = { a_prev_pad[M].begin() + horiz_start, a_prev_pad[M].begin() + horiz_end };
                        }

                        for (int L = vert_start; L < vert_end; L++) {
                            for (int M = horiz_start; M < horiz_end; M++) {
                                for (int P = 0; P < n_C_prev; P++) {
                                    // Update gradients for the windowand the filter's parameters using the code formulas given above                        
                                    da_prev_pad[L][M][P] += W[L][M][P][c] * dZ[i][h][w][c];
                                    dW[L][M][P][c] += a_slice[L][M][P] * dZ[i][h][w][c];
                                    db[L][M][P][c] += dZ[i][h][w][c];
                                }
                            }
                        }
                    } // end of loop channel c
                }
            }
            // Set the ith training example's dA_prev to the unpadded da_prev_pad
            for (int h = 0; h < n_H_prev; h++) {
                for (int w = 0; w < n_W_prev; w++) {
                    for (int c = 0; c < n_C_prev; c++) { //channels        
                        dA_prev[i][h][w][c] = da_prev_pad[h + n_pad][w + n_pad][c];
                    }
                }
            }
        }  //end the loop of sample i
        return make_tuple(dA_prev, dW, db);
    }

    vector<vector<bool>> create_mask_from_window(v2f X) {
        int row = X.size(), col = X[0].size();
        float max_val = get_max_of_2dv (X);
        vector<vector<bool>> mask(row, v1f(col, false));
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (X[i][j] == max_val) {
                    mask[i][j] = true;
                }
            }
        }
        return mask;
    }


    v2f distribute_value(float dz, tuple <int, int> shape) {  
        int n_H = get<0>(shape), n_W = get<1>(shape);
        v2f a(n_H, v1f(n_W, 0));
        int size = n_H * n_W;

        for (int i = 0; i < n_H; i++) {
            for (int j = 0; i < n_W; j++) {
                a[i][j] *= dz / size;
            }
        }
        return a;
    }

    template <typename T>
    v4f pool_backward(v4f dA, tuple <v4f, unordered_map <string, v4f>> cache, string mode = "max") {
        v4f A_prev = get<0>(cache); //activation of previous layer
        int m = A_prev.size(), n_H_prev = A_prev[0].size(), n_W_prev = A_prev[1].size(), n_C_prev = A_prev[2].size();
        v4f dA_prev (m, v3f (n_H_prev, v2f (n_W_prev, v1f(n_C_prev, 0))));
        int m_dA = dA.size(), n_H = dA[0].size(), n_W = dA[1].size(), n_C = dA[2].size();
        if (m != m_dA) {
            cout << "Error: batch sample size doesn't match between two layers" << endl;
            return;
        }
        unordered_map <string, T> hparameters = get<1>(cache);
        int stride = hparameters["stride"]; //stride size
        int f = hparameters["f"];  //filter size

        //dA_prev: gradient of cost wrt Activation of previous layer
        v4f dA_prev(m, v3f(n_H_prev, v2f(n_W_prev, v1f(n_C_prev, 0))));
        for (int i = 0; i < m; i++) { // # loop over batch examples
            //# select training example from A_prev(˜1 line)
            v3f a_prev = { A_prev[i].begin(), A_prev[i].end() };
            for (int h = 0; h < n_H; h++) { // : # loop on the vertical axis
                for (int w = 0; w < n_W; w++) { //# loop on the horizontal axis
                    for (int c = 0; c < n_C; c++) { //: # loop over the channels(depth)
                        // # Find the corners of the current "slice" (˜4 lines)
                        int vert_start = h * stride;
                        int vert_end = h * stride + f;
                        int horiz_start = w * stride;
                        int horiz_end = w * stride + f;

                        //vector<float>::const_iterator vert_start = myVec.begin() + h * stride;
                        //vector<float>::const_iterator vert_end = myVec.begin() + h * stride + f;


                        //vector<float>::const_iterator horiz_start = myVec.begin() + w * stride;
                        //vector<float>::const_iterator horiz_end = myVec.begin() + w * stride + f;

                        // # Compute the backward propagation in both modes.
                        if (mode == "max") {
                            //# Use the corners and "c" to define the current slice from a_prev(˜1 line)
                                //v2f a_prev_slice(, v1f);  = a_prev [vert_start:vert_end, horiz_start : horiz_end, c]

                            v2f a_prev_slice(vert_end - vert_start, v1f(horiz_end - horiz_start, 0));
                            for (int L = vert_start; L < vert_end; L++) {
                                for (int M = 0; M < horiz_end; M++) {
                                    a_prev_slice[L][M] = A_prev[L][M][c];

                                }
                            }
                            vector<vector<bool>> mask = create_mask_from_window(a_prev_slice);
                            for (int L = vert_start; L < vert_end; L++) {
                                for (int M = 0; M < horiz_end; M++) {
                                    dA_prev[i][L][M][c] += mask[L][M] * dA[i][h][w][c];
                                }
                            }
                        }

                        else if (mode == "average") {
                            float da = dA[i][h][w][c];
                            tuple <int, int> shape = (f, f);
                            v2f dv = distribute_value(da, shape);
                            for (int L = vert_start; L < vert_end; L++) {
                                for (int M = 0; M < horiz_end; M++) {
                                    dA_prev[i][L][M][c] += dv[L][M];
                                }
                            }
                        }
                        // size check
                        if (dA_prev.size() != A_prev.size() || dA_prev[0].size() != A_prev[0].size() || dA_prev[1].size() != A_prev[1].size() || dA_prev[2].size() != A_prev[2].size())
                            cout << "Error: size of gradeint_activation doens't match size of activation" << endl;
                        return dA_prev;
                    }
                }
            }
        }
    }
};
