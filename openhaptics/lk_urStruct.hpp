#include <math.h>
#include <stdio.h>
#include <algorithm>

struct Quaternion
{
    double w, x, y, z;
};

struct EulerAngles {
    double roll, pitch, yaw;
};

EulerAngles ToEulerAngles(Quaternion q) {
    EulerAngles angles;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        angles.pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles.pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}

Quaternion ToQuaternion(double yaw, double pitch, double roll) // yaw (Z), pitch (Y), roll (X)
{
    // Abbreviations for the various angular functions
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

struct transform_M
{   
    //total_element
    double element[16];
    int size=16;
    //n-vector
    double& r11 = *(element+0);
    double& nx = *(element+0);
    double& r21 = *(element+4);
    double& ny = *(element+4);
    double& r31 = *(element+8);
    double& nz = *(element+8);
    double& wn = *(element+12);

    //o-vector
    double& r12 = *(element+1);
    double& ox = *(element+1);
    double& r22 = *(element+5);
    double& oy = *(element+5);
    double& r32 = *(element+9);
    double& oz = *(element+9);
    double& wo = *(element+13);

    //a-vector
    double& r13 = *(element+2);
    double& ax = *(element+2);
    double& r23 = *(element+6);
    double& ay = *(element+6);
    double& r33 = *(element+10);
    double& az = *(element+10);
    double& wa = *(element+14);

    //pose-vector
    double& x = *(element+3);//(Unit: mm)
    double& y = *(element+7);//(Unit: mm)
    double& z = *(element+11);//(Unit: mm)
    double& w = *(element+15);

    struct Vector4
    {
        double element[4];
        int size=4;
        double& X = *(element+0);
        double& Y = *(element+1);
        double& Z = *(element+2);
        double& W = *(element+3);
        
        transform_M::Vector4& operator=(const transform_M::Vector4& other)//copy
        {
            if(&other!=this)
            {
                std::copy(other.element, other.element + other.size, element);
            }
            return *this;
        }
    
        transform_M::Vector4& operator*(const transform_M& other)//matmul
        {
            transform_M::Vector4& new_transform = *new transform_M::Vector4;
        
            for(int c =0;c<4;c++)
            {
                new_transform.element[c]=0;

                for(int e=0;e<4;e++)
                {
                    new_transform.element[c]+=element[e]*other.element[(e*4)+c];
                }
            }
            return new_transform;
        }

    };


    void setOnes()
    {
        r11=1;r12=1;r13=1;x=1;
        r21=1;r22=1;r23=1;y=1;
        r31=1;r32=1;r33=1;z=1;
         wn=1; wo=1; wa=1;w=1;
    }

    void setZeros()
    {
        r11=0;r12=0;r13=0;x=0;
        r21=0;r22=0;r23=0;y=0;
        r31=0;r32=0;r33=0;z=0;
         wn=0; wo=0; wa=0;w=0;
    }

    void setIdentity()
    {
        r11=1;r12=0;r13=0;x=0;
        r21=0;r22=1;r23=0;y=0;
        r31=0;r32=0;r33=1;z=0;
         wn=0; wo=0; wa=0;w=1;
    }

    double determinant()
    {
        // r11 r12 r13 x
        // r21 r22 r23 y
        // r31 r32 r33 z
        //  wn  wo  wa w
        // base calculate 3x3 Matrix determinant form (()*(()-())) - (()*(()-())) + (()*(()-()))
        return       +r11*(  ((r22)*((r33*w)-(z*wa))) - ((r23)*((r32*w)-(z*wo))) + ((y)*((r32*wa)-(r33*wo)))  )
                     -r12*(  ((r21)*((r33*w)-(z*wa))) - ((r23)*((r31*w)-(z*wn))) + ((y)*((r31*wa)-(r33*wn)))  )
                     +r13*(  ((r21)*((r32*w)-(z*wo))) - ((r22)*((r31*w)-(z*wn))) + ((y)*((r31*wo)-(r32*wn)))  )
                     -  x*(  ((r21)*((r32*wa)-(r33*wo))) - ((r22)*((r31*wa)-(r33*wn))) + ((r23)*((r31*wo)-(r32*wn))) );
    }
    
    transform_M& inverse()
    {
        //ref: This was lifted from MESA implementation of the GLU library.

        double inv[16], det;
        double* m = this->element;

        inv[ 0] =  m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
        inv[ 4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
        inv[ 8] =  m[4] * m[ 9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[ 9];
        inv[12] = -m[4] * m[ 9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[ 9];
        inv[ 1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
        inv[ 5] =  m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
        inv[ 9] = -m[0] * m[ 9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[ 9];
        inv[13] =  m[0] * m[ 9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[ 9];
        inv[ 2] =  m[1] * m[ 6] * m[15] - m[1] * m[ 7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[ 7] - m[13] * m[3] * m[ 6];
        inv[ 6] = -m[0] * m[ 6] * m[15] + m[0] * m[ 7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[ 7] + m[12] * m[3] * m[ 6];
        inv[10] =  m[0] * m[ 5] * m[15] - m[0] * m[ 7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[ 7] - m[12] * m[3] * m[ 5];
        inv[14] = -m[0] * m[ 5] * m[14] + m[0] * m[ 6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[ 6] + m[12] * m[2] * m[ 5];
        inv[ 3] = -m[1] * m[ 6] * m[11] + m[1] * m[ 7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[ 9] * m[2] * m[ 7] + m[ 9] * m[3] * m[ 6];
        inv[ 7] =  m[0] * m[ 6] * m[11] - m[0] * m[ 7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[ 8] * m[2] * m[ 7] - m[ 8] * m[3] * m[ 6];
        inv[11] = -m[0] * m[ 5] * m[11] + m[0] * m[ 7] * m[ 9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[ 9] - m[ 8] * m[1] * m[ 7] + m[ 8] * m[3] * m[ 5];
        inv[15] =  m[0] * m[ 5] * m[10] - m[0] * m[ 6] * m[ 9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[ 9] + m[ 8] * m[1] * m[ 6] - m[ 8] * m[2] * m[ 5];
 
        det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
 
        if(det == 0)
        {
            printf("det == 0\n");
            return *this;
        }
            
 
        det = 1.f / det;

        transform_M& new_transform = *new transform_M;

        for(int e = 0; e < 16; e++)
        {
            new_transform.element[e] = inv[e] * det;
        }
        

        return new_transform;
    }

    
    transform_M& operator=(const transform_M& other)//copy
    {
        if(&other!=this)
        {
            std::copy(other.element, other.element + other.size, element);
        }
        return *this;
    }

    transform_M& operator*(const transform_M& other)//matmul
    {
        transform_M& new_transform = *new transform_M;
    
        for(int c =0;c<4;c++)
        {
            for(int r =0;r<4;r++)
            {
                new_transform.element[c+(r*4)]=0;
                for(int e=0;e<4;e++)
                {
                    new_transform.element[c+(r*4)]+=element[e+(r*4)]*other.element[(e*4)+c];
                }
            }
        }

        return new_transform;
    }
    
    
    Vector4& operator*(const Vector4& other)//matmul
    {
        Vector4& new_transform = *new Vector4;
    
        for(int c =0;c<4;c++)
        {
            new_transform.element[c]=0;

            for(int e=0;e<4;e++)
            {
                new_transform.element[c]+=element[e+(c*4)]*other.element[e];
            }
        }
        return new_transform;
    }
    
};

struct UR_Joint_Position
{
    double element[6];
    int size=6;
    double& base_rad = *(element+0);//(Unit: rad)
    double& shoulder_rad = *(element+1);//(Unit: rad)
    double& elbow_rad = *(element+2);//(Unit: rad)
    double& wrist1_rad = *(element+3);//(Unit: rad)
    double& wrist2_rad = *(element+4);//(Unit: rad)
    double& wrist3_rad = *(element+5);//(Unit: rad)
    UR_Joint_Position& operator=(const UR_Joint_Position& other)
    {
        if(&other!=this)
        {
            std::copy(other.element, other.element + other.size, element);
        }
        return *this;
    }

    void setZeros()
    {
        base_rad=0.0;
        shoulder_rad=0.0;
        elbow_rad=0.0;
        wrist1_rad=0.0;
        wrist2_rad=0.0;
        wrist3_rad=0.0;
    }
};

struct UR_RPY_Position
{
    double element[6];
    int size=6;
    double& X = *(element+0);//(Unit: mm)
    double& Y = *(element+1);//(Unit: mm)
    double& Z = *(element+2);//(Unit: mm)
    double& Roll = *(element+3);//(Unit: rad)
    double& Pitch = *(element+4);//(Unit: rad)
    double& Yaw = *(element+5);//(Unit: rad)

    UR_RPY_Position& operator=(const UR_RPY_Position& other)
    {
        if(&other!=this)
        {
            std::copy(other.element, other.element + other.size, element);
        }
        return *this;
    }

    void setZeros()
    {
        X=0.0;
        Y=0.0;
        Z=0.0;
        Roll=0.0;
        Pitch=0.0;
        Yaw=0.0;
    }
    
};

struct UR_Euler_Position
{
    double element[6];
    int size=6;
    double& X = *(element+0);//(Unit: mm)
    double& Y = *(element+1);//(Unit: mm)
    double& Z = *(element+2);//(Unit: mm)
    double& Roll = *(element+3);//(Unit: rad)
    double& Pitch = *(element+4);//(Unit: rad)
    double& Yaw = *(element+5);//(Unit: rad)

    UR_Euler_Position& operator=(const UR_Euler_Position& other)
    {
        if(&other!=this)
        {
            std::copy(other.element, other.element + other.size, element);
        }
        return *this;
    }

    void setZeros()
    {
        X=0.0;
        Y=0.0;
        Z=0.0;
        Roll=0.0;
        Pitch=0.0;
        Yaw=0.0;
    }
    
};

struct UR_Clip_Vector3
{
    double element[6];
    int size=6;
    double& X_Min = *(element+0);
    double& X_Max = *(element+1);
    double& Y_Min = *(element+2);
    double& Y_Max = *(element+3);
    double& Z_Min = *(element+4);
    double& Z_Max = *(element+5);

    UR_Clip_Vector3& operator=(const UR_Clip_Vector3& other)
    {
        if(&other!=this)
        {
            std::copy(other.element, other.element + other.size, element);
        }
        return *this;
    }

    void setZeros()
    {
        X_Min=0.0;
        X_Max=0.0;
        Y_Min=0.0;
        Y_Max=0.0;
        Z_Min=0.0;
        Z_Max=0.0;
    }
};


struct UR_Clip_RPY
{
    double element[12];
    int size=12;
    double& X_Min = *(element+0);
    double& X_Max = *(element+1);
    double& Y_Min = *(element+2);
    double& Y_Max = *(element+3);
    double& Z_Min = *(element+4);
    double& Z_Max = *(element+5);
    double& Roll_Min = *(element+6);
    double& Roll_Max = *(element+7);
    double& Pitch_Min = *(element+8);
    double& Pitch_Max = *(element+9);
    double& Yaw_Min = *(element+10);
    double& Yaw_Max = *(element+11);

    UR_Clip_RPY& operator=(const UR_Clip_RPY& other)
    {
        if(&other!=this)
        {
            std::copy(other.element, other.element + other.size, element);
        }
        return *this;
    }

    void setZeros()
    {
        X_Min=0.0;
        X_Max=0.0;
        Y_Min=0.0;
        Y_Max=0.0;
        Z_Min=0.0;
        Z_Max=0.0;

        Roll_Min=0.0;
        Roll_Max=0.0;
        Pitch_Min=0.0;
        Pitch_Max=0.0;
        Yaw_Min=0.0;
        Yaw_Max=0.0;
    }
};


struct UR_Joint_Current
{
    double element[6];
    int size=6;
    double& base_A = *(element+0);//(Unit: A)
    double& shoulder_A = *(element+1);//(Unit: A)
    double& elbow_A = *(element+2);//(Unit: A)
    double& wrist1_A = *(element+3);//(Unit: A)
    double& wrist2_A = *(element+4);//(Unit: A)
    double& wrist3_A = *(element+5);//(Unit: A)

    UR_Joint_Current& operator=(const UR_Joint_Current& other)
    {
        if(&other!=this)
        {
            std::copy(other.element, other.element + other.size, element);
        }
        return *this;
    }
    void setZeros()
    {
        base_A=0.0;
        shoulder_A=0.0;
        elbow_A=0.0;
        wrist1_A=0.0;
        wrist2_A=0.0;
        wrist3_A=0.0;
    }
};

struct UR_TCP_Position
{
    double element[6];
    int size=6;
    double& X = *(element+0);//(Unit: mm)
    double& Y = *(element+1);//(Unit: mm)
    double& Z = *(element+2);//(Unit: mm)
    double& RX = *(element+3);//(Unit: rad)
    double& RY = *(element+4);//(Unit: rad)
    double& RZ = *(element+5);//(Unit: rad)
    
    UR_TCP_Position& operator=(const UR_TCP_Position& other)
    {
        if(&other!=this)
        {
            std::copy(other.element, other.element + other.size, element);
        }
        return *this;
    }

    UR_TCP_Position& operator+(const UR_TCP_Position& other)
    {
        UR_TCP_Position& new_UR_TCP_Position = *new UR_TCP_Position;

        for(int e = 0; e < size; e++)
        {
            new_UR_TCP_Position.element[e]=element[e]+other.element[e];
        }
        
        return new_UR_TCP_Position;
    }
    
    UR_TCP_Position& operator-(const UR_TCP_Position& other)
    {
        UR_TCP_Position& new_UR_TCP_Position = *new UR_TCP_Position;

        for(int e = 0; e < size; e++)
        {
            new_UR_TCP_Position.element[e]=element[e]-other.element[e];
        }
        
        return new_UR_TCP_Position;
    }

    UR_TCP_Position& operator*(const UR_TCP_Position& other)
    {
        UR_TCP_Position& new_UR_TCP_Position = *new UR_TCP_Position;

        for(int e = 0; e < size; e++)
        {
            new_UR_TCP_Position.element[e]=element[e]*other.element[e];
        }
        
        return new_UR_TCP_Position;
    }

    UR_TCP_Position& operator*(const double& other)
    {
        UR_TCP_Position& new_UR_TCP_Position = *new UR_TCP_Position;

        for(int e = 0; e < size; e++)
        {
            new_UR_TCP_Position.element[e]=element[e]*other;
        }
        return new_UR_TCP_Position;
    }

    UR_TCP_Position& operator/(const UR_TCP_Position& other)
    {
        UR_TCP_Position& new_UR_TCP_Position = *new UR_TCP_Position;

        for(int e = 0; e < size; e++)
        {
            new_UR_TCP_Position.element[e]=element[e]/other.element[e];
        }
        
        return new_UR_TCP_Position;
    }

    void operator+=(const UR_TCP_Position& other)
    {
        for(int e = 0; e < size; e++)
        {
            element[e]+=other.element[e];
        }
    }

    void operator-=(const UR_TCP_Position& other)
    {
        for(int e = 0; e < size; e++)
        {
            element[e]-=other.element[e];
        }
    }

    void operator*=(const UR_TCP_Position& other)
    {
        for(int e = 0; e < size; e++)
        {
            element[e]*=other.element[e];
        }
    }

    void operator/=(const UR_TCP_Position& other)
    {
        for(int e = 0; e < size; e++)
        {
            element[e]/=other.element[e];
        }
    }

    void setZeros()
    {
        for(int e = 0; e < size; e++)
        {
            element[e]=0.0;
        }
    }
};