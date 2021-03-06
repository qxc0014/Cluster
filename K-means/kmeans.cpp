#include<iostream>
#include<fstream>
#include<vector>
#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<Eigen/Core>
using namespace std;
using namespace cv;
const float min_extent = 1;
const float com_float = 2e6;
const int k = 3;//聚类成2
string points_list = "/home/esoman/c++code/c++/K-means/varied.txt";
class QuadNode{
    public:
        QuadNode(){}
        QuadNode(QuadNode* child_root,Point2d center,float extent,vector<int> points_index,bool is_leaf)
        :extent_(extent),points_index_(points_index),is_leaf_(is_leaf),center_(center){
           // for(int i=0;i<8;i++)
           //    child_root_[i] = new Octant();
        }
        int depth_=0;//节点的深度
        QuadNode* child_root_[8]={nullptr,nullptr,nullptr,nullptr};//存四个子立方体的指针
        Point2d center_;//当前立方体的中心坐标
        float extent_;//当前正方体的半边长
        vector<int> points_index_;//当前立方体的包含点的Index
        bool is_leaf_;//当前坐标是否为叶子

};
class distindex{
    public:
        distindex(float dist_,int index_):dist(dist_),index(index_){}
        float dist;
        int index;
};
class result{
    public:
    result(float worst_dis_):worst_dis(worst_dis_){}//用于搜索一个近邻点
    result(float worst_dis_,int k):worst_dis(worst_dis_),worst_dis_cap(vector<distindex>(k,distindex(worst_dis_,-1))),size(k){ 
    }
    float worst_dis=0;
    int index;
    int num=0;
    int size;
    vector<distindex> worst_dis_cap;
    void add_point(float bias,int node_index);
};
void result::add_point(float bias,int node_index){
        if(num != this->size) num++;//已插入值的个数
        if(bias >= worst_dis_cap[this->size-1].dist) return;//大于最大值直接跳出
        int i = num-1;//已经插入最大值的index
        while(i>0){
             if(bias < worst_dis_cap[i-1].dist){
                this->worst_dis_cap[i] = worst_dis_cap[i-1];
                i--;
             }else{
                break;
             }
        }
        worst_dis_cap[i].dist = bias;
        worst_dis_cap[i].index = node_index;
        this->worst_dis =  worst_dis_cap[this->size-1].dist;
}
QuadNode* build_Quadtree(QuadNode* root,vector<Point2d>* db,Point2d center,float extent,vector<int> points_index){
    if(points_index.size() == 0) {
        return nullptr;
    }
    if(root == nullptr){
       // cout << "节点深度：" << depth << "节点宽度" << width << endl;
        root = new QuadNode(nullptr,center,extent,points_index,true);
    }
    if(extent < min_extent && points_index.size()<=1){
        root->is_leaf_ = true;//叶子节点
    }else{
        root->is_leaf_ = false;//不是叶子
        vector<vector<int>> child_point_index(4);
        for(auto point_idx:points_index){
            int Coordinate = 0;
            if((*db)[point_idx].x > center.x){
                Coordinate = Coordinate | 1;
            }
            if((*db)[point_idx].y > center.y){
                Coordinate = Coordinate | 2;
            }
            child_point_index[Coordinate].push_back(point_idx);
        }
        float factor[2] = {-0.5,0.5};
        vector<Point2d> child_center(4);
        float child_extent=0;
        for(int i = 0;i < 4;i++){
            child_center[i].x = center.x + factor[(i&1)>0]*extent;
            child_center[i].y = center.y + factor[(i&2)>0]*extent;
            child_extent = 0.5 *extent;
            //cout << child_extent << endl;
            root->child_root_[i] = build_Quadtree(root->child_root_[i],db,child_center[i],child_extent,child_point_index[i]);
        }   
    }
    return root;
} 
//判断球与立方体的方位
bool overlap(QuadNode* root,Point2d Point,float worst_dis){
    //分三种情况:
    //第一种:球与立方体没有接触,只要投影的某个方向满足就可以
    float xyz[2];
    xyz[0] = fabs(root->center_.x - Point.x);
    xyz[1] = fabs(root->center_.y - Point.y);
    float max_dis = (root->extent_+ worst_dis);
    if( xyz[0] > max_dis || xyz[1] > max_dis ) return false;
    //第二种:球与立方体相交（通过投影判断）至少有两个投影面包含了圆心就可以认为是相交
    if(((xyz[0]<root->extent_)+(xyz[1]<root->extent_))>=1) return true;
    //第三种:补充第二种，在边界处相交不满足第二种
    float x = (xyz[0]-root->extent_)>0?(xyz[0]-root->extent_):0;
    float y = (xyz[1]-root->extent_)>0?(xyz[1]-root->extent_):0;
    if(x*x+y*y<worst_dis*worst_dis) return true;
}
//判断球是否在立方体内
bool inside(QuadNode* root,Point2d Point,float worst_dis){
    float xyz[2];
    xyz[0] = fabs(root->center_.x - Point.x);
    xyz[1] = fabs(root->center_.y - Point.y);
    float max_dis = (root->extent_ - worst_dis);
    return ((xyz[0] < max_dis) && (xyz[1] < max_dis));

}
bool Quadtree_knn_search(QuadNode* root,vector<Point2d>* db,Point2d Point,result &a){
    //先判断当前root是否为空指针
    if(root == nullptr) return false;
    //判断当前的节点是否为叶子
    if((root->is_leaf_ == true) && root->points_index_.size() == 1){
       //计算worst_dis
       //cout << "找到叶子！" << endl;
       Eigen::Vector2d radius(Point.x - (*db)[root->points_index_[0]].x,
                              Point.y - (*db)[root->points_index_[0]].y);
       float dis = radius.squaredNorm();
       a.add_point(dis,root->points_index_[0]);
       //a.worst_dis = a.worst_dis < dis? a.worst_dis:dis;
       //判断现在的球是否在立方体内，如果在可以提前终止
       bool q = inside(root,Point,a.worst_dis);
      // cout << a.worst_dis_cap[0].dist << endl;
       return q;
    }
    //判断目标点所属象限
    int Coordinate = 0;
    if(Point.x > root->center_.x){
        Coordinate = Coordinate | 1;
    }
    if(Point.y > root->center_.y){
        Coordinate = Coordinate | 2;
    }
        //迭代寻找新的子象限
    if(Quadtree_knn_search(root->child_root_[Coordinate],db,Point,a)) return true;
    //当发现最近的子象限都不能完全包裹最坏距离，那么就要扫描其他的子象限
    for(int i = 0;i<4;i++){
        //先排除刚才已经扫描过的象限
        if(i == Coordinate || root->child_root_[i] == nullptr) continue;
        //再排除球与立方体不相交的情况
        //cout << i << endl;
        if(false == overlap(root->child_root_[i],Point,a.worst_dis)) continue;
        //最后对这个象限进行计算worst_dis
        if(Quadtree_knn_search(root->child_root_[i],db,Point,a)) return true;
    }

    //再次判断现在的球是否在立方体内，如果在可以提前终止
    return inside(root,Point,a.worst_dis);
}
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
        points.x = a*10+300;
		p = strtok(NULL, split);
        b = atof(p);
        points.y = b*10+300;
        datasets.push_back(points);
    }
    //绘制所有数据点
    Mat image(600, 600, CV_8UC3, cv::Scalar(255, 255, 255));
    for(int i =0;i<datasets.size();i++){
        cv::circle(image,datasets[i],1,cv::Scalar(255, 255, 255));
    }
    //k-means初始化
    //随机选取两个点作为初始中心点
    int center_index[k];
    srand(time(0));
    center_index[0] = rand()%datasets.size();
    while((center_index[1] = rand()%datasets.size())==center_index[0]){
    cout << "1" << center_index[1] << endl;
    cout << "2"<< center_index[2] << endl;
    }
    vector<Point2d> center={datasets[center_index[0]],datasets[center_index[1]],datasets[center_index[2]]};
    vector<Point2d> last_center;
    //cv::circle(image,center[0],2,cv::Scalar(0, 0, 255));//红色为初始中心点
    //cv::circle(image,center[1],2,cv::Scalar(255, 0, 0));
    float x_min,x_max,y_min,y_max;
    x_min = com_float;
    x_max = -com_float;
    y_min = com_float;
    y_max = -com_float;
    vector<int> point_index(center.size());
    std::partial_sum(point_index.begin(), point_index.end(), point_index.begin(), [](const int&a, int b) {return a + 1;});
    for(auto point:center){
        x_min = x_min < point.x?x_min:point.x;
        x_max = x_max > point.x?x_max:point.x;
        y_min = y_min < point.y?y_min:point.y;
        y_max = y_max > point.y?y_max:point.y;
    }
    Point2d Quadtree_center_point((x_min+x_max)/2.,(y_min+y_max)/2.);
    float extent = (x_max-x_min)>(y_max-y_min)?(x_max-x_min):(y_max-y_min);
    extent = ceil(0.5*extent);
    QuadNode* root=nullptr;
    root = build_Quadtree(root,&center,Quadtree_center_point,extent,point_index);
    //遍历每个点找最近的中心点
    vector<vector<Point2d>> last_kmean_class(k);//上次的k聚类中的点集
    while(1){
        vector<vector<Point2d>> kmean_class(k);//存放不同类的点的index
        for(int i=0;i<datasets.size();i++){
            result a(2e6,2);
            Quadtree_knn_search(root,&center,datasets[i],a);
            if(a.worst_dis_cap[0].index == 0){
                kmean_class[0].push_back(datasets[i]);
                cv::circle(image,datasets[i],4,cv::Scalar(0, 0, 255));//红色
            }else if(a.worst_dis_cap[0].index == 1){
                kmean_class[1].push_back(datasets[i]);
                cv::circle(image,datasets[i],4,cv::Scalar(255, 0, 0));//红色
            }else if(a.worst_dis_cap[0].index == 2){
                kmean_class[2].push_back(datasets[i]);
                cv::circle(image,datasets[i],4,cv::Scalar(0, 255, 0));//红色
            }
        //cv::circle(image,datasets[i],2,cv::Scalar(0, 0, 255));
        }
        cout << "class1:" << kmean_class[0] << endl;
        cout << "class2:" << kmean_class[1] << endl;    
        for(int i=0;i<k;i++){
            Point2d sum(0,0);
            for(auto point_:kmean_class[i])
                sum +=point_;
            center[i] = sum / (int)kmean_class[i].size();
        }
        if(kmean_class == last_kmean_class && center == last_center){//中心点不变、K个类中点不变就停止
           // break;
        }
        cv::circle(image,center[0],10,cv::Scalar(0, 0, 255));
        cv::circle(image,center[1],10,cv::Scalar(255, 0, 0));
        imshow("kmeans",image);
        cv::circle(image,center[0],10,cv::Scalar(255, 255, 255));//消除痕迹
        cv::circle(image,center[1],10,cv::Scalar(255, 255, 255));
        waitKey();
        last_kmean_class = kmean_class;
        last_center = center;
    }
    //显示
    return 0;
}
