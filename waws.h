/*****************************************************************************/
/*                                                                           */
/*****************************************************************************/
#ifndef WAWS_H_
#define WAWS_H_
#include<iostream>
#include<cmath>
#include "Eigen/Dense"
#include "Eigen/Cholesky"

// struct Point
// {
//     double x, y, z;
// };
class Point
{
    private:
        double x_, y_, z_;

    public:
        Point();
        Point(double x, double y, double z=0.0);
        Point(const Point &p);
        virtual ~Point();
        double x() const;
        double y() const;
        double z() const;
        friend Point operator+(const Point &p1, const Point &p2);
        friend Point operator-(const Point &p1, const Point &p2);
        friend bool operator==(const Point &p1, const Point &p2);
        friend bool operator!=(const Point &p1, const Point &p2);
        friend std::ostream & operator<<(std::ostream &out, const Point &p);
        friend std::istream & operator>>(std::istream &in, Point &p);
};


double davenportSpectrum(double v10, double I10, double z, double alpha);
double coherence();
double turubulenceIntensity();


#endif