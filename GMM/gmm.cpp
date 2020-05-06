#include<iostream>
#include<fstream>
#include<vector>
#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<Eigen/Eigenvalues>
using namespace std;
using namespace cv;
const float min_extent = 1;
const float com_float = 2e6;
const int img_size=400;
const int k = 3;//聚类成2
int point_num =1500;
string points_list = "/home/esoman/c++code/c++/GMM/blobs.txt";
class Gauss{
    public:
        Point2d center;//均值
        Eigen::Matrix2d covMat;//方差
        double pai_k;//模型的权重vector
        Eigen::MatrixXd gauss_result;
        Eigen::MatrixXd p_z_x;
        double Nk;
        long double cal_p(Point2d point){
            double gailv=0;
            Eigen::Vector2d error(point.x-this->center.x,point.y-this->center.y);
            //double w = -0.5*error.transpose()*(covMat.inverse())*error;
           // cout <<"w="<< w <<endl;
            gailv  = (1./sqrt(2 * M_PI * covMat.determinant()))*exp(-0.5*error.transpose()*(covMat.inverse())*error);
            return gailv;
        }
};
int main(int argc, char const *argv[])
{
    //读取数据
    ifstream fin;
    fin.open(points_list,ios::in);
    string buff;
    vector<Point2d> datasets;
    while (getline(fin, buff))
    {
        Point2d points;
        char *s_input = (char *)buff.c_str();
        const char * split = ",";
        char *p = strtok(s_input, split);
        double a,b;
		a = atof(p);
        points.x = a*10+img_size/2;//需要根据对应的点云调整大小
		p = strtok(NULL, split);
        b = atof(p);
        points.y = b*10+img_size/2;
        datasets.push_back(points);
    }
    //绘制所有数据点
    Mat image(img_size, img_size, CV_8UC3, cv::Scalar(255, 255, 255));
    for(int i =0;i<datasets.size();i++){
        cv::circle(image,datasets[i],1,cv::Scalar(0, 0, 0));
    }
    //gmm初始化
    //随机选取两个点作为初始miu
    int center_index[k];
    srand(time(0));
    center_index[0] = rand()%datasets.size();
    while((center_index[1] = rand()%datasets.size())==center_index[0]){
    }
    vector<Gauss> gauss_dis(k); 
    for(int i = 0;i<k;i++){
        gauss_dis[i].center = datasets[center_index[i]];
        gauss_dis[i].covMat = 8*Eigen::Matrix2d::Identity(2,2);
        gauss_dis[i].pai_k = 1. / (double)k;
    }
           // cout <<p_x<< endl;

    while(1){
        //E-step
        
        Eigen::MatrixX3d point_p_z_x(datasets.size(),k);
        for(int j=0;j<datasets.size();j++){
            long double p_x=0;
            for(int i=0;i<k;i++){
                p_x += gauss_dis[i].pai_k*gauss_dis[i].cal_p(datasets[j]);
            }        
            //cout <<"px="<< p_x<< endl;
            for(int i=0;i<k;i++){
                double p_z_x=0;
                if(p_x != 0){
                p_z_x = gauss_dis[i].pai_k*gauss_dis[i].cal_p(datasets[j]) / (long double)p_x;
                point_p_z_x(j,i) = p_z_x;
                }

            }
            circle(image,datasets[j],2,cv::Scalar(255*point_p_z_x(j,0), 255*point_p_z_x(j,1), 255*point_p_z_x(j,2)));
        }
            //M-step
        //计算Nk
        //cout <<"pzx:"<< point_p_z_x<<endl;
        for(int i =0;i<k;i++){
           // cout << gauss_dis[i].covMat.determinant() << endl;
           // circle(image, gauss_dis[i].center,gauss_dis[i].covMat.determinant(),cv::Scalar(0, 0, 255));
            gauss_dis[i].Nk = point_p_z_x.col(i).sum();
            cout <<"Nk" << gauss_dis[i].Nk  << endl;
            //circle(image, gauss_dis[i].center,gauss_dis[i].covMat.determinant(),cv::Scalar(0, 0, 255));
           // gauss_dis[i].covMat.eigenvalues() ;
            //Size size(gauss_dis[i].covMat.eigenvalues,gauss_dis[i].covMat.eigenvalues[1]);
            //cvEllipse(&image,gauss_dis[i].center,size,2*M_PI,M_PI,-M_PI,cv::Scalar(0, 0, 255));
        }
        for(int i = 0;i<k;i++){
            Point2d sum_point(0,0);
            Eigen::Matrix2d sum_covMat(Eigen::Matrix2d::Zero());
            for(int j =0;j<datasets.size();j++){    
                sum_point += point_p_z_x(j,i)*datasets[j];
            }
            gauss_dis[i].center = sum_point/gauss_dis[i].Nk;
            for(int j =0;j<datasets.size();j++){   
                Eigen::Vector2d error_cov(datasets[j].x-gauss_dis[i].center.x,datasets[j].y-gauss_dis[i].center.y);
                sum_covMat += point_p_z_x(j,i)*error_cov*error_cov.transpose();
            }
            gauss_dis[i].covMat = sum_covMat/(double)gauss_dis[i].Nk;
            gauss_dis[i].pai_k = gauss_dis[i].Nk/datasets.size();
        }
        imshow("gmm",image);
        waitKey();
    }
    //显示
   /* for(int i =0;i<1500;i++){
        if(gauss_dis[0].gauss_result(i,0)>0.01){
            circle(image,datasets[i],2,cv::Scalar(0, 0, 255));
        }
    }*/
    return 0;
}
