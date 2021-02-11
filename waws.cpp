#include "waws.h"
using namespace std;


/*****************************************************************************/
/*                                   Point                                   */
/*****************************************************************************/
Point::Point()
{

}

Point::Point(double x, double y, double z)
{
    x_ = x;
    y_ = y;
    z_ = z;
}

Point::Point(const Point &p)
{
    this->x_ = p.x_;
    this->y_ = p.y_;
    this->z_ = p.z_;
}

Point::~Point()
{

}

double Point::x() const
{
    return x_;
}

double Point::y() const
{
    return y_;
}

double Point::z() const
{
    return z_;
}

Point operator+(const Point &p1, const Point &p2)
{
    Point p;
    p.x_ = p1.x_ + p2.x_;
    p.y_ = p1.y_ + p2.y_;
    p.z_ = p1.z_ + p2.z_;
    return p;
}

Point operator-(const Point &p1, const Point &p2)
{
    Point p;
    p.x_ = p1.x_ - p2.x_;
    p.y_ = p1.y_ - p2.y_;
    p.z_ = p1.z_ - p2.z_;
    return p;
}

bool operator==(const Point &p1, const Point &p2)
{
    bool cond1 = abs(p1.x_ - p2.x_) < 1e-10;
    bool cond2 = abs(p1.y_ - p2.y_) < 1e-10;
    bool cond3 = abs(p1.z_ - p2.z_) < 1e-10;
    return cond1 && cond2 && cond3;
}

bool operator!=(const Point &p1, const Point &p2)
{
    bool cond1 = abs(p1.x_ - p2.x_) > 1e-10;
    bool cond2 = abs(p1.y_ - p2.y_) > 1e-10;
    bool cond3 = abs(p1.z_ - p2.z_) > 1e-10;
    return cond1 || cond2 || cond3;
}

ostream & operator<<(ostream &out, const Point &p)
{
    out << p.x_ << ", " << p.y_ << ", " << p.z_;
    return out;
}

istream & operator>>(istream &in, Point &p)
{
    in >> p.x_ >> p.y_ >> p.z_;
    return in;
}


int main()
{
    Point p1 = Point(1, 2, 3);
    Point p2 = Point(2, 3, 4);
    Point p3 = p1 + p2;
    Point p4 = p1 - p2;
    cout << " ----- --------" << endl;
    cout << p1 << endl << p2 << endl << p3 << endl << p4 << endl;

    Point p5;
    cout << " ----- --------" << endl;
    cout << "Enter x, y, z: " << endl;
    cin >> p5;
    cout << p5 << endl;

    Point p6 = p5;
    Point p7 = Point(p5);
    bool b1 = p6 == p5;
    bool b2 = p7 == p5;
    bool b3 = p6 == p7;
    cout << " ----- --------" << endl;
    cout << b1 << ", " << b2 << ", " << b3 << endl;
}
