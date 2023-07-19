/*
*   Author      : LEEKW (Kang-Won Lee)
*   E-mail      : kangwon353@dongguk.edu
*   Version     : 1.0.1
*   Data        : 2019-11-13
*   Brief       : 
*/

#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

#include <math.h>
#include <iomanip>
#include <sstream>

// < define : UR modbus setting >
// (ref : https://www.universal-robots.com/how-tos-and-faqs/how-to/ur-how-tos/modbus-server-16377/)
// modbus config : joint angle (rad)
#define BASE        "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x0e\x00\x01"
#define SHOULDER    "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x0f\x00\x01"
#define ELBOW       "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x10\x00\x01"
#define WRIST1      "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x11\x00\x01"
#define WRIST2      "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x12\x00\x01"
#define WRIST3      "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x13\x00\x01"
// modbus config : joint current (mA)
#define BASE_CURRENT        "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x22\x00\x01"
#define SHOULDER_CURRENT    "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x23\x00\x01"
#define ELBOW_CURRENT       "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x24\x00\x01"
#define WRIST1_CURRENT      "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x25\x00\x01"
#define WRIST2_CURRENT      "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x26\x00\x01"
#define WRIST3_CURRENT      "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x27\x00\x01"
// modbus config : TCP position (mm, mrad)
#define TCP_X   "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x90\x00\x01"
#define TCP_Y   "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x91\x00\x01"
#define TCP_Z   "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x92\x00\x01"
#define TCP_RX  "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x93\x00\x01"
#define TCP_RY  "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x94\x00\x01"
#define TCP_RZ  "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x95\x00\x01"
// < /define : UR modbus setting >

// define : PI & 2PI
#define LK_PI   3.1415926535897932384626433832795
#define LK_2PI  6.283185307179586476925286766559

enum UR_MODBUS_HEX_UNIT{THOUSAND,TEN,PM_THOUSAND,PM_TEN};

struct transform_matrix {

    // < config : matrix elements >
    double element[16];
    int size    = 16;
    // n-vector
    double& r11 = *(element+0);
    double& r21 = *(element+4);
    double& r31 = *(element+8);
    double& nx  = *(element+0);
    double& ny  = *(element+4);
    double& nz  = *(element+8);
    double& nw  = *(element+12);
    // o-vector
    double& r12 = *(element+1);
    double& r22 = *(element+5);
    double& r32 = *(element+9);
    double& ox  = *(element+1);
    double& oy  = *(element+5);
    double& oz  = *(element+9);
    double& ow  = *(element+13);
    // a-vector
    double& r13 = *(element+2);
    double& r23 = *(element+6);
    double& r33 = *(element+10);
    double& ax  = *(element+2);
    double& ay  = *(element+6);
    double& az  = *(element+10);
    double& aw  = *(element+14);
    // position-vector
    double& px  = *(element+3);
    double& py  = *(element+7);
    double& pz  = *(element+11);
    double& pw  = *(element+15);
    // < /config : matrix elements >

    struct Vector4 {

        double element[4];
        int size = 4;
        double& X   = *(element+0);
        double& Y   = *(element+1);
        double& Z   = *(element+2);
        double& W   = *(element+3);

        // copy the element
        transform_matrix::Vector4& operator=(const transform_matrix::Vector4& other) {
            
            if(&other!=this) {
                
                std::copy(other.element, other.element + other.size, element);
            }

            return *this;
        }

        // matrix multiplication
        transform_matrix::Vector4& operator*(const transform_matrix& other) {

            transform_matrix::Vector4& new_transform = *new transform_matrix::Vector4;

            for(int i = 0; i < 4; i++) {

                new_transform.element[i] = 0;

                for(int j = 0; j < 4; j++) {

                    new_transform.element[i] += element[j] * other.element[(j*4)+i];
                }
            }

            return new_transform;
        }
    };

    void setOnes() {

        r11 = 1; r12 = 1; r13 = 1; px = 1;
        r21 = 1; r22 = 1; r23 = 1; py = 1;
        r31 = 1; r32 = 1; r33 = 1; pz = 1;
        nw  = 1; ow  = 1; aw  = 1; pw = 1;
    }

    void setZeros() {

        r11 = 0; r12 = 0; r13 = 0; px = 0;
        r21 = 0; r22 = 0; r23 = 0; py = 0;
        r31 = 0; r32 = 0; r33 = 0; pz = 0;
        nw  = 0; ow  = 0; aw  = 0; pw = 0;
    }

    void setIdentity() {

        r11 = 1; r12 = 0; r13 = 0; px = 0;
        r21 = 0; r22 = 1; r23 = 0; py = 0;
        r31 = 0; r32 = 0; r33 = 1; pz = 0;
        nw  = 0; ow  = 0; aw  = 0; pw = 1;
    }

    double getDeterminet() {

        // r11 r12 r13 px
        // r21 r22 r23 py
        // r31 r32 r33 pz
        //  nw  ow  aw pw

        double determinet   = 0;

        determinet  = +r11*(  ((r22)*((r33*pw)-(pz*aw))) - ((r23)*((r32*pw)-(pz*ow))) + ((py)*((r32*aw)-(r33*ow)))  )
                      -r12*(  ((r21)*((r33*pw)-(pz*aw))) - ((r23)*((r31*pw)-(pz*nw))) + ((py)*((r31*aw)-(r33*nw)))  )
                      +r13*(  ((r21)*((r32*pw)-(pz*ow))) - ((r22)*((r31*pw)-(pz*nw))) + ((py)*((r31*ow)-(r32*nw)))  )
                      - px*(  ((r21)*((r32*aw)-(r33*ow))) - ((r22)*((r31*aw)-(r33*nw))) + ((r23)*((r31*ow)-(r32*nw))) );

        return determinet;
    }

    transform_matrix& getInverse() {

        // ref : MESA implementation of the GLU library

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

        if (det == 0) {
            
            printf("FATALL ERROR : (Det = 0)\n");
            return *this;
        }

        det = 1.f / det;

        transform_matrix& new_transform = *new transform_matrix;

        for (int i = 0; i < 16; i++) {

            new_transform.element[i] = inv[i] * det;
        }

        return new_transform;
    }
    
    // copy the element
    transform_matrix& operator=(const transform_matrix& other) {

        if (&other != this) {

            std::copy(other.element, other.element + other.size, element);
        }

        return *this;
    }

    // matrix multiplication
    transform_matrix& operator*(const transform_matrix& other) {

        transform_matrix& new_transform = *new transform_matrix;

        for (int i = 0; i < 4; i++) {

            for (int j = 0; j < 4; j++) {

                new_transform.element[i+(j*4)] = 0;

                for (int k = 0; k < 4; k++) {

                    new_transform.element[i+(j*4)] += element[k+(j*4)] * other.element[(k*4)+i];
                }
            }
        }

        return new_transform;
    }

    Vector4& operator*(const Vector4& other) {

        Vector4& new_transform = *new Vector4;

        for (int i = 0; i < 4; i++) {

            new_transform.element[i] = 0;

            for (int j = 0; j < 4; j++) {

                new_transform.element[i] += element[j+(i*4)] * other.element[j];
            }
        }

        return new_transform;
    }
};

struct UR_jointPosition {

    double element[6];
    int size = 6;

    // ur robot joint position (unit : rad)
    double& base        = *(element + 0);
    double& shoulder    = *(element + 1);
    double& elbow       = *(element + 2);
    double& wrist1      = *(element + 3);
    double& wrist2      = *(element + 4);
    double& wrist3      = *(element + 5);

    UR_jointPosition& operator=(const UR_jointPosition& other) {

        if (&other != this) {

            std::copy(other.element, other.element + other.size, element);
        }

        return *this;
    }

    void setZeros() {

        base        = 0.0;
        shoulder    = 0.0;
        elbow       = 0.0;
        wrist1      = 0.0;
        wrist2      = 0.0;
        wrist3      = 0.0;
    }
};

struct UR_rpyPosition {

    double element[6];
    int size = 6;

    // ur robot rpy position (unit : mm, rad)
    double& X       = *(element + 0);
    double& Y       = *(element + 1);
    double& Z       = *(element + 2);
    double& Roll    = *(element + 3);
    double& Pitch   = *(element + 4);
    double& Yaw     = *(element + 5);

    UR_rpyPosition& operator=(const UR_rpyPosition& other) {

        if(&other != this) {

            std::copy(other.element, other.element + other.size, element);
        }

        return *this;
    }

    void setZeros() {

        X       = 0.0;
        Y       = 0.0;
        Z       = 0.0;
        Roll    = 0.0;
        Pitch   = 0.0;
        Yaw     = 0.0;
    }
};

struct UR_jointCurrent {

    double element[6];
    int size = 6;

    // ur robot joint current (unit : mA)
    double& base        = *(element + 0);
    double& shoulder    = *(element + 1);
    double& elbow       = *(element + 2);
    double& wrist1      = *(element + 3);
    double& wrist2      = *(element + 4);
    double& wrist3      = *(element + 5);

    UR_jointCurrent& operator=(const UR_jointCurrent& other) {

        if (&other != this) {

            std::copy(other.element, other.element + other.size, element);
        }

        return *this;
    }

    void setZeros() {

        base        = 0.0;
        shoulder    = 0.0;
        elbow       = 0.0;
        wrist1      = 0.0;
        wrist2      = 0.0;
        wrist3      = 0.0;
    }
};

struct UR_tcpPosition {

    double element[6];
    int size = 6;

    // ur robot tcp position (unit : mm, rad)
    double& X   = *(element + 0);
    double& Y   = *(element + 1);
    double& Z   = *(element + 2);
    double& RX  = *(element + 3);
    double& RY  = *(element + 4);
    double& RZ  = *(element + 5);

    UR_tcpPosition& operator=(const UR_tcpPosition& other) {

        if (&other != this) {

            std::copy(other.element, other.element + other.size, element);
        }

        return *this;
    }

    UR_tcpPosition& operator+(const UR_tcpPosition& other) {

        UR_tcpPosition& new_UR_tcpPosition = *new UR_tcpPosition;

        for (int i = 0; i < size; i++) {

            new_UR_tcpPosition.element[i] = element[i] + other.element[i];
        }

        return new_UR_tcpPosition;
    }

    UR_tcpPosition& operator-(const UR_tcpPosition& other) {

        UR_tcpPosition& new_UR_tcpPosition = *new UR_tcpPosition;

        for (int i = 0; i < size; i++) {

            new_UR_tcpPosition.element[i] = element[i] - other.element[i];
        }

        return new_UR_tcpPosition;
    }

    UR_tcpPosition& operator*(const UR_tcpPosition& other) {

        UR_tcpPosition& new_UR_tcpPosition = *new UR_tcpPosition;

        for (int i = 0; i < size; i++) {

            new_UR_tcpPosition.element[i] = element[i] * other.element[i];
        }

        return new_UR_tcpPosition;
    }

    UR_tcpPosition& operator/(const UR_tcpPosition& other) {

        UR_tcpPosition& new_UR_tcpPosition = *new UR_tcpPosition;

        for (int i = 0; i < size; i++) {

            new_UR_tcpPosition.element[i] = element[i] / other.element[i];
        }

        return new_UR_tcpPosition;
    }

    void operator+=(const UR_tcpPosition& other) {

        for (int i = 0; i < size; i++) {

            element[i] += other.element[i];
        }
    }

    void operator-=(const UR_tcpPosition& other) {

        for (int i = 0; i < size; i++) {

            element[i] -= other.element[i];
        }
    }

    void operator*=(const UR_tcpPosition& other) {

        for (int i = 0; i < size; i++) {

            element[i] *= other.element[i];
        }
    }

    void operator/=(const UR_tcpPosition& other) {

        for (int i = 0; i < size; i++) {

            element[i] /= other.element[i];
        }
    }

    void setZeros() {

        for (int i = 0; i < size; i++) {

            element[i] = 0;
        }
    }
};

class LK_UR5e {

    // < Robot Information >
    private:
        // UR5e DH-parameter (unit : mm) 
        // ref : https://www.universal-robots.com/how-tos-and-faqs/faq/ur-faq/parameters-for-calculations-of-kinematics-and-dynamics-45257/
        const double UR5e_d1 =  0.1625*1000;
        const double UR5e_a2 = -0.425*1000;
        const double UR5e_a3 = -0.3922*1000;
        const double UR5e_d4 =  0.1333*1000;
        const double UR5e_d5 =  0.0997*1000;
        const double UR5e_d6 =  0.0996*1000;

        // UR10 DH-parameter (unit : mm) 
        // ref : https://www.universal-robots.com/how-tos-and-faqs/faq/ur-faq/parameters-for-calculations-of-kinematics-and-dynamics-45257/
        const double UR10_d1 =  0.1273*1000;
        const double UR10_a2 = -0.612*1000;
        const double UR10_a3 = -0.5723*1000;
        const double UR10_d4 =  0.1639*1000;
        const double UR10_d5 =  0.1157*1000;
        const double UR10_d6 =  0.0922*1000;

        // Robot control state
        const int communication_wateTime    = 2; // (unit : ms)
        const double ControlerBox_version   = 5.4;
        const unsigned int MULT_JOINTSTATE_ = 1000000;
    // < /Robot Information >

    private:
        // config : Ethernet socket
        int Ethernet_socket = 0;
        int Ethernet_valread;
        struct sockaddr_in Ethernet_serv_addr;
        char Ethernet_buffer[1024] = {0};
        bool is_Ethernet_connect = false;

    // < Ethernet >
    public:
        bool Ethernet_connect(const char* ip_addr, int port);
        void Ethernet_setMovej(UR_tcpPosition* tcp_position, float max_vel, float max_acc);
        void Ethernet_setMovel(UR_tcpPosition* tcp_position, float max_vel, float max_acc);
        void Ethernet_setServoj(UR_tcpPosition* tcp_position, float timeout);
        void Ethernet_close();
    // < /Ethernet >

    // < Position Control >
    public:
        bool UR_serverOpen(const char* ip_addr, int port);
        bool UR_serverProcess();
        bool UR_serverClose();
        void UR_control(UR_tcpPosition* tcp_position, int on_off);
        bool UR_uploadScript(const char* myServer_ipAddr, int myServer_port, const char* controler_ipAddr, int controler_port);
    private:
        bool is_Ethernet_realtimeServer_open = false;
        UR_tcpPosition realtime_tcpPosition;
        bool first_realtime_tcpPosition_send = true;
        int UR_sockfd_;
        int myServer_sockfd_;
        int keepalive = 1;
    // < /Position Control >

    // < calculation >
    public:
        void get_forward(UR_jointPosition* input_jointPosition, transform_matrix* output_matrix);
        void get_DHmatrix(transform_matrix* output_matrix, double DH_theta, double DH_a, double DH_d, double DH_alpha);
        void get_transform2tcpPosition(transform_matrix* input_transform, UR_tcpPosition* output_tcpPosition);
        void get_transform2rpyPosition(transform_matrix* input_transform, UR_rpyPosition* output_rpyPosition);
        void get_tcpPosition2transform(UR_tcpPosition* input_tcpPosition, transform_matrix* output_transform);
        void get_rpyPosition2transform(UR_rpyPosition* input_rpyPosition, transform_matrix* output_transform);
        bool check_tcpPosition_boundry(UR_tcpPosition& ref_position, UR_tcpPosition& target_position, double errorPos, double errorAng);
        void matmul(double* front_matrix, double* back_marix, double* output_matrix);
    // < /calculation >

    // < socket modbus communication >
    public:
        bool modbus_connect(const char* ip_addr, int port);
        void modbus_get_tcpPosition(UR_tcpPosition* tcp_position);
        void modbus_get_jointPosition(UR_jointPosition* joint_position);
        void modbus_get_jointCurrent(UR_jointCurrent* joint_current);
        void modbus_close();
    // < /socket modbus communication >

    // < modbus communication >
    private:
        int modbus_socket = 1;
        int modbus_valread;
        struct sockaddr_in modbus_servAddr;
        bool is_modbus_connect = false;
        uint8_t modbus_buffer[1024] = {0x00,};
        float modbus_hex2value_f(uint8_t* buf, UR_MODBUS_HEX_UNIT unit);
        double modbus_hex2value_d(uint8_t* buf, UR_MODBUS_HEX_UNIT unit);
        bool get_modbus(const char* REGISTERS, float* getvalue, UR_MODBUS_HEX_UNIT type);
        bool get_modbus(const char* REGISTERS, double* getvalue, UR_MODBUS_HEX_UNIT type);
    // < /modbus communication >
};

// <  define : Ethernet >
bool LK_UR5e::Ethernet_connect(const char* ip_addr, int port) {

    // default setting
    if (ip_addr == NULL) {

        ip_addr = "127.0.0.1";
    }
    if (port == 0) {

        port = 30003;
    }

    if ( (Ethernet_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {

        printf("[ERROR] Failed to create a Ethernet socket! \n");
        return false;
    }

    Ethernet_serv_addr.sin_family   = AF_INET;
    Ethernet_serv_addr.sin_port     = htons(port);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if (inet_pton(AF_INET, ip_addr, &Ethernet_serv_addr.sin_addr) <= 0) {

        printf("[ERROR] Failed to convert IP address! \n");
        return false;
    }

    if (connect(Ethernet_socket, (struct sockaddr *)&Ethernet_serv_addr, sizeof(Ethernet_serv_addr)) < 0) {

        printf("[ERROR] Failed to connect a Ethernet socket! \n");
        return false;
    }

    is_Ethernet_connect = true;

    return is_Ethernet_connect;
}

void LK_UR5e::Ethernet_setMovej(UR_tcpPosition* tcp_position, float max_vel, float max_acc) {

    if (is_Ethernet_connect == false || is_Ethernet_realtimeServer_open == true) {return;}

    std::stringstream stream;
    stream <<"movej(p[" << std::fixed<<std::setprecision(6) <<tcp_position->X/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->Y/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->Z/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RX<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RY<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RZ<<"],"
    << "a="<< std::fixed<< std::setprecision(3) <<max_acc<<","
    << "v="<< std::fixed<< std::setprecision(3) <<max_vel<<")\n";

    std::string command = stream.str();
    char const *pchar   = command.c_str();

    send(Ethernet_socket, pchar, command.length(), 0);
    std::cout << pchar<<std::endl;
}

void LK_UR5e::Ethernet_setMovel(UR_tcpPosition* tcp_position, float max_vel, float max_acc) {

    if(is_Ethernet_connect == false || is_Ethernet_realtimeServer_open == true){return;}

    std::stringstream stream;
    stream <<"movel(p[" << std::fixed<<std::setprecision(6) <<tcp_position->X/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->Y/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->Z/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RX<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RY<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RZ<<"],"
    << "a="<< std::fixed<< std::setprecision(3) <<max_acc<<","
    << "v="<< std::fixed<< std::setprecision(3) <<max_vel<<")\n";
    
    std::string command = stream.str();    
    char const *pchar   = command.c_str();
    

    send(Ethernet_socket, pchar, command.length(), 0);
    std::cout << pchar<<std::endl;
}

void LK_UR5e::Ethernet_setServoj(UR_tcpPosition* tcp_position, float timeout) {

    if (is_Ethernet_connect == false || is_Ethernet_realtimeServer_open == true) {return;}

    std::stringstream stream;
    stream <<"servoj(get_inverse_kin(p[" << std::fixed<<std::setprecision(6) <<tcp_position->X/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->Y/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->Z/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RX<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RY<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RZ<<"]),"
    << "t="<< std::fixed<< std::setprecision(4) <<timeout<<")\n";

    std::string commnad = stream.str();
    char const *pchar   = commnad.c_str();
    send(Ethernet_socket, pchar, commnad.length(), 0);
    std::cout << pchar<<std::endl;    
}

void LK_UR5e::Ethernet_close() {

    if (is_Ethernet_connect == false) {return;}
    close(Ethernet_socket);
}
// < /define : Ethernet >

// < define : Position Control >
bool LK_UR5e::UR_serverOpen(const char* ip_addr, int port) {

    if (is_Ethernet_realtimeServer_open == true) {return false;}

    char buffer[256];
    struct sockaddr_in serv_addr;
    struct sockaddr_in cli_addr;
    socklen_t clilen;

    int n;
    int flag;

    UR_sockfd_       = -1;
    myServer_sockfd_ = socket(AF_INET, SOCK_STREAM, 0);

    if (myServer_sockfd_ < 0) {

        printf("[ERROR] opening socket for reverse communication! \n");
        return false;
    }

    bzero((u_char *) &serv_addr, sizeof(serv_addr));
    
    serv_addr.sin_family        = AF_INET;
    serv_addr.sin_addr.s_addr   = INADDR_ANY;
    serv_addr.sin_port          = htons(port);
    flag = 1;

    setsockopt(myServer_sockfd_, IPPROTO_TCP, TCP_NODELAY, (u_char * ) &flag, sizeof(int));
    setsockopt(myServer_sockfd_, IPPROTO_TCP, TCP_QUICKACK, (char *) &flag, sizeof(flag));
    setsockopt(myServer_sockfd_, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(int));

    if (bind(myServer_sockfd_, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {

        printf("[ERROR] binding socket for reverse communication! \n");
        return false;
    }
    listen(myServer_sockfd_, 5);

    clilen = sizeof(cli_addr);
    UR_sockfd_ = accept(myServer_sockfd_, (struct sockaddr *) &cli_addr, &clilen);

    if (UR_sockfd_ < 0) {

        printf("[ERROR] accepting reverse communication! \n");
        return false;
    }

    printf("[INFO] UR has been successfully connected to server \n");
    is_Ethernet_realtimeServer_open = true;

    return is_Ethernet_realtimeServer_open;
}

bool LK_UR5e::UR_serverProcess() {

    if (is_Ethernet_realtimeServer_open == false) {

        printf("[ERROR] UR_server did not open, please check the connect status! \n");
        return false;
    }

    if (first_realtime_tcpPosition_send == true) {

        first_realtime_tcpPosition_send = false;

        if (is_modbus_connect == true) {

            modbus_get_tcpPosition(&realtime_tcpPosition);
        }
        else {

            return false;
        }
    }

    unsigned int bytes_written;
    int tmp;
    unsigned char buf[28];

    for (int i = 0; i < 6; i++) {

        if (i < 3) {

            tmp = htonl((int)(realtime_tcpPosition.element[i] * (MULT_JOINTSTATE_ / 1000.0)));
        }
        else {

            tmp = htonl((int)(realtime_tcpPosition.element[i] * MULT_JOINTSTATE_));
        }

        buf[i * 4]      = tmp & 0xff;
        buf[i * 4 + 1]  = (tmp >> 8) & 0xff;
        buf[i * 4 + 2]  = (tmp >> 16) & 0xff;
        buf[i * 4 + 3]  = (tmp >> 24) & 0xff;
    }

    tmp = htonl((int) keepalive);
    buf[6 * 4]      = tmp & 0xff;
    buf[6 * 4 + 1]  = (tmp >> 8) & 0xff;
    buf[6 * 4 + 2]  = (tmp >> 16) & 0xff;
    buf[6 * 4 + 3]  = (tmp >> 24) & 0xff;

    bytes_written = write(UR_sockfd_, buf, 28);

    return true;
}

bool LK_UR5e::UR_serverClose() {

    if (is_Ethernet_realtimeServer_open == false) {

        return false;
    }

    UR_control(&realtime_tcpPosition, 0);
    UR_serverProcess();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    is_Ethernet_realtimeServer_open = false;
    close(UR_sockfd_);
    close(myServer_sockfd_);

    return true;
}

void LK_UR5e::UR_control(UR_tcpPosition* tcp_position, int on_off) {

    if (is_Ethernet_realtimeServer_open == false) {return;}

    realtime_tcpPosition = (*tcp_position);
    keepalive = on_off;
}

bool LK_UR5e::UR_uploadScript(const char* myServer_ipAddr, int myServer_port, const char* controler_ipAddr, int controler_port) {

    if (is_Ethernet_connect == false) {Ethernet_connect(controler_ipAddr, controler_port);}
    if (is_Ethernet_connect == false) {printf("[ERROR] Ethernet is not connected to UR controler! \n"); return false;}
    if (is_modbus_connect == false) {modbus_connect(controler_ipAddr, 502);}
    if (is_modbus_connect == false) {printf("[ERROR] Not connected to UR modbus! \n"); return false;}

    float servoj_time              = 0.008f;    // time (s), range (0.002 ~ infinite)
    float servoj_lookahead_time    = 0.1f;      // time (s), range (0.03 ~ 0.2) : smoothens the trajectory with this lookahead time
    float servoj_gain              = 300.0f;

    // < UR script >
    std::string cmd_str;
    char buf[128];

    cmd_str = "def driverProg():\n";

    sprintf(buf, "\tMULT_jointstate = %i\n", MULT_JOINTSTATE_);
	cmd_str += buf;
	cmd_str += "\tSERVO_IDLE = 0\n";
	cmd_str += "\tSERVO_RUNNING = 1\n";
	cmd_str += "\tcmd_servo_state = SERVO_IDLE\n";
    cmd_str += "\tcmd_servo_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"; //get_actual_joint_positions()
	cmd_str += "\tdef set_servo_setpoint(q):\n";
	cmd_str += "\t\tenter_critical\n";
	cmd_str += "\t\tcmd_servo_state = SERVO_RUNNING\n";
	cmd_str += "\t\tcmd_servo_q = get_inverse_kin(q)\n";//<=this script and privous modern driver different 
	cmd_str += "\t\texit_critical\n";
	cmd_str += "\tend\n";
	cmd_str += "\tthread servoThread():\n";
	cmd_str += "\t\tstate = SERVO_IDLE\n";
	cmd_str += "\t\twhile True:\n";
	cmd_str += "\t\t\tenter_critical\n";
	cmd_str += "\t\t\tq = cmd_servo_q\n";
	cmd_str += "\t\t\tdo_brake = False\n";
	cmd_str += "\t\t\tif (state == SERVO_RUNNING) and ";
	cmd_str += "(cmd_servo_state == SERVO_IDLE):\n";
	cmd_str += "\t\t\t\tdo_brake = True\n";
	cmd_str += "\t\t\tend\n";
	cmd_str += "\t\t\tstate = cmd_servo_state\n";
	cmd_str += "\t\t\tcmd_servo_state = SERVO_IDLE\n";
	cmd_str += "\t\t\texit_critical\n";
	cmd_str += "\t\t\tif do_brake:\n";
	cmd_str += "\t\t\t\tstopj(1.0)\n";
	cmd_str += "\t\t\t\tsync()\n";
	cmd_str += "\t\t\telif state == SERVO_RUNNING:\n";

    if (ControlerBox_version >= 3.1) {

        sprintf(buf, "\t\t\t\tservoj(q, t=%.4f, lookahead_time=%.4f, gain=%.0f)\n", servoj_time, servoj_lookahead_time, servoj_gain);
    }
    else {

        sprintf(buf, "\t\t\t\tservoj(q, t=%.4f)\n", servoj_time);
    }

    cmd_str += buf;

    cmd_str += "\t\t\telse:\n";
	cmd_str += "\t\t\t\tsync()\n";
	cmd_str += "\t\t\tend\n";
	cmd_str += "\t\tend\n";
	cmd_str += "\tend\n";

    sprintf(buf, "\tsocket_open(\"%s\", %i)\n", myServer_ipAddr, myServer_port);
	cmd_str += buf;

    cmd_str += "\tthread_servo = run servoThread()\n";
	cmd_str += "\tkeepalive = 1\n";
	cmd_str += "\twhile keepalive > 0:\n";
	cmd_str += "\t\tparams_mult = socket_read_binary_integer(6+1)\n";
	cmd_str += "\t\tif params_mult[0] > 0:\n";
	cmd_str += "\t\t\tq = p[params_mult[1] / MULT_jointstate, ";
	cmd_str += "params_mult[2] / MULT_jointstate, ";
	cmd_str += "params_mult[3] / MULT_jointstate, ";
	cmd_str += "params_mult[4] / MULT_jointstate, ";
	cmd_str += "params_mult[5] / MULT_jointstate, ";
	cmd_str += "params_mult[6] / MULT_jointstate]\n";
	cmd_str += "\t\t\tkeepalive = params_mult[7]\n";
	cmd_str += "\t\t\tset_servo_setpoint(q)\n";
	cmd_str += "\t\tend\n";
	cmd_str += "\tend\n";
	cmd_str += "\tsleep(1)\n";
	cmd_str += "\tsocket_close()\n";
	cmd_str += "\tkill thread_servo\n";
	cmd_str += "end\n";
    // < /UR script >

    char const *pchar = cmd_str.c_str();

    send(Ethernet_socket, pchar, cmd_str.length(), 0);
    printf("[INFO] UR script has been successfully uploaded. \n");

    return true;
}
// < /define : Position Control >

// < define : calculation >
void LK_UR5e::get_forward(UR_jointPosition* input_jointPosition, transform_matrix* output_matrix){

    transform_matrix T1; get_DHmatrix(&T1, input_jointPosition->base, 0, UR5e_d1, LK_PI/2);
    transform_matrix T2; get_DHmatrix(&T2, input_jointPosition->shoulder, UR5e_a2, 0, 0);
    transform_matrix T3; get_DHmatrix(&T3, input_jointPosition->elbow, UR5e_a3, 0, 0);
    transform_matrix T4; get_DHmatrix(&T4, input_jointPosition->wrist1, 0, UR5e_d4, LK_PI/2);
    transform_matrix T5; get_DHmatrix(&T5, input_jointPosition->wrist2, 0, UR5e_d5, -LK_PI/2);
    transform_matrix T6; get_DHmatrix(&T6, input_jointPosition->wrist3, 0, UR5e_d6, 0);

    (*output_matrix) = T1 * T2 * T3 * T4 * T5 * T6;
}

void LK_UR5e::get_DHmatrix(transform_matrix* output_matrix, double DH_theta, double DH_a, double DH_d, double DH_alpha) {

    output_matrix->r11  =  cos(DH_theta);
    output_matrix->r12  = -sin(DH_theta) * cos(DH_alpha);
    output_matrix->r13  =  sin(DH_theta) * sin(DH_alpha);
    output_matrix->px   =  DH_a * cos(DH_theta);

    output_matrix->r21  =  sin(DH_theta);
    output_matrix->r22  =  cos(DH_theta) * cos(DH_alpha);
    output_matrix->r23  = -cos(DH_theta) * sin(DH_alpha);
    output_matrix->py   =  DH_a * cos(DH_theta);

    output_matrix->r31  =  0;
    output_matrix->r32  =  sin(DH_alpha);
    output_matrix->r33  =  cos(DH_alpha);
    output_matrix->pz   =  DH_d;

    output_matrix->nw   =  0;
    output_matrix->ow   =  0;
    output_matrix->aw   =  0;
    output_matrix->pw   =  1;
}

void LK_UR5e::get_transform2tcpPosition(transform_matrix* input_transform, UR_tcpPosition* output_tcpPosition) {

    // transfer 'rotation matrix' to 'tcp position (UR)'
    // using "Equivalent angle axis" (ref : "introduction to robotics - John j.Craig" book 73p)

    output_tcpPosition->X   = input_transform->px;
    output_tcpPosition->Y   = input_transform->py;
    output_tcpPosition->Z   = input_transform->pz;

    double theta = acos( (input_transform->r11 + input_transform->r22 + input_transform->r33 - 1) / 2.0);

    // robot unstable exception
    if (theta < 0.00001 && theta > -0.00001) {

        output_tcpPosition->RX  = 0.0;
        output_tcpPosition->RY  = 0.0;
        output_tcpPosition->RZ  = 0.0;

        return;
    }

    double temp_K = (1.0 / (2*sin(theta)));

    double RX   = temp_K * (input_transform->r32 - input_transform->r23);
    double RY   = temp_K * (input_transform->r13 - input_transform->r31);
    double RZ   = temp_K * (input_transform->r21 - input_transform->r12);

    // robot unstable exception
    double vector_norm  = sqrt( (RX*RX) + (RY*RY) + (RZ*RZ) );

    output_tcpPosition->RX  = theta * (RX / vector_norm);
    output_tcpPosition->RY  = theta * (RY / vector_norm);
    output_tcpPosition->RZ  = theta * (RZ / vector_norm);
}

void LK_UR5e::get_transform2rpyPosition(transform_matrix* input_transform, UR_rpyPosition* output_rpyPosition) {

    // transfer 'Roll, Pitch, Yaw (RPY) angle' to 'rotation matrix'
    // (ref : "introdution to robotics - SAEEDD B,NIKU" book 67p")

    output_rpyPosition->X = input_transform->px;
    output_rpyPosition->Y = input_transform->py;
    output_rpyPosition->Z = input_transform->pz;

    output_rpyPosition->Roll    = atan2( input_transform->ny,
                                         input_transform->nx);
    
    output_rpyPosition->Pitch   = atan2( -input_transform->nz,
                                         ( (input_transform->nx * cos(output_rpyPosition->Roll)) + (input_transform->ny * sin(output_rpyPosition->Roll)) ) );

    output_rpyPosition->Yaw     = atan2( ( -(input_transform->ay * cos(output_rpyPosition->Roll)) + (input_transform->ax * sin(output_rpyPosition->Roll)) ),
                                         (  (input_transform->oy * cos(output_rpyPosition->Roll)) - (input_transform->ox * sin(output_rpyPosition->Roll)) ) );
}

void LK_UR5e::get_tcpPosition2transform(UR_tcpPosition* input_tcpPosition, transform_matrix* output_transform) {

    // transfer 'rotation matrix' to 'tcp position (UR)'
    // using "Equivalent angle axis" (ref : "introduction to robotics - John j.Craig" book 73p)

    output_transform->px = input_tcpPosition->X;
    output_transform->py = input_tcpPosition->Y;
    output_transform->pz = input_tcpPosition->Z;
    output_transform->pw = 1;
    output_transform->nw = 0;
    output_transform->ow = 0;
    output_transform->aw = 0;

    double kx   = input_tcpPosition->RX;
    double ky   = input_tcpPosition->RY;
    double kz   = input_tcpPosition->RZ;

    double theta = sqrt( (kx*kx)+(ky*ky)+(kz*kz) );

    if (theta < 0.00001 && theta > -0.00001) {

        output_transform->r11   = 1;
        output_transform->r12   = 0;
        output_transform->r13   = 0;

        output_transform->r21   = 0;
        output_transform->r22   = 1;
        output_transform->r23   = 0;

        output_transform->r31   = 0;
        output_transform->r32   = 0;
        output_transform->r33   = 1;

        return;
    }

    kx  = kx/theta;
    ky  = ky/theta;
    kz  = kz/theta;

    double ctheta   = cos(theta);
    double stheta   = sin(theta);
    double vtheta   = 1.0 - cos(theta);

    output_transform->r11   = (kx * kx * vtheta) + (ctheta);
    output_transform->r12   = (kx * ky * vtheta) - (kz * stheta);
    output_transform->r13   = (kx * kz * vtheta) + (ky * stheta);

    output_transform->r21   = (ky * kx * vtheta) + (kz * stheta);
    output_transform->r22   = (ky * ky * vtheta) + (ctheta);
    output_transform->r23   = (ky * kz * vtheta) - (kx * stheta);

    output_transform->r31   = (kx * kz * vtheta) - (ky * stheta);
    output_transform->r32   = (ky * kz * vtheta) + (kx * stheta);
    output_transform->r33   = (kz * kz * vtheta) + (ctheta);
}

void LK_UR5e::get_rpyPosition2transform(UR_rpyPosition* input_rpyPosition, transform_matrix* output_transform) {

    // transfer 'Roll, Pitch, Yaw (RPY) angle' to 'rotation matrix'
    // (ref : "introdution to robotics - SAEEDD B,NIKU" book 67p")

    output_transform->px = input_rpyPosition->X;
    output_transform->py = input_rpyPosition->Y;
    output_transform->pz = input_rpyPosition->Z;

    double pi_a = input_rpyPosition->Roll;
    double pi_o = input_rpyPosition->Pitch;
    double pi_n = input_rpyPosition->Yaw;

    output_transform->r11 = cos(pi_a) * cos(pi_o);
    output_transform->r12 = cos(pi_a) * sin(pi_o) * sin(pi_n) - sin(pi_a) * cos(pi_n);
    output_transform->r13 = cos(pi_a) * sin(pi_o) * cos(pi_n) + sin(pi_a) * sin(pi_n);

    output_transform->r21 = sin(pi_a) * cos(pi_o);
    output_transform->r22 = sin(pi_a) * sin(pi_o) * sin(pi_n) + cos(pi_a) * cos(pi_n);
    output_transform->r23 = sin(pi_a) * sin(pi_o) * cos(pi_n) - cos(pi_a) * sin(pi_n);

    output_transform->r31 = -sin(pi_o);
    output_transform->r32 =  cos(pi_o) * sin(pi_n);
    output_transform->r33 =  cos(pi_o) * cos(pi_n);

    output_transform->nw = 0;
    output_transform->ow = 0;
    output_transform->aw = 0;
    output_transform->pw = 1;
}

bool LK_UR5e::check_tcpPosition_boundry(UR_tcpPosition& ref_position, UR_tcpPosition& target_position, double errorPos = 0.05, double errorAng = 0.05) {

    return (ref_position.X > (target_position.X - errorPos) && ref_position.X < (target_position.X + errorPos))
        && (ref_position.Y > (target_position.Y - errorPos) && ref_position.Y < (target_position.Y + errorPos))
        && (ref_position.Z > (target_position.Z - errorPos) && ref_position.Z < (target_position.Z + errorPos))
        && (ref_position.RX > (target_position.RX - errorAng) && ref_position.RX < (target_position.RX + errorAng))
        && (ref_position.RY > (target_position.RY - errorAng) && ref_position.RY < (target_position.RY + errorAng))
        && (ref_position.RZ > (target_position.RZ - errorAng) && ref_position.RZ < (target_position.RZ + errorAng));
}

void LK_UR5e::matmul(double* front_matrix, double* back_marix, double* output_matrix) {

    for (int i = 0; i < 4; i++) {

        for (int j = 0; j < 4; j++) {

            output_matrix[i + (j*4)] = 0;

            for (int k = 0; k < 4; k++) {

                output_matrix[i + (j*4)] += front_matrix[k + (j*4)] * back_marix[(k*4) + i];
            }
        }
    }
}
// < /define : calculation >

// < define : socket modbus communication >
bool LK_UR5e::modbus_connect(const char* ip_addr, int port) {

    // default setting
    if (ip_addr == NULL) {

        ip_addr = "127.0.0.1";
    }
    if (port == 0) {

        port = 502;
    }

    if ( (modbus_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {

        printf("[ERROR] Failed to create a modbus socket! \n");
        return false;
    }

    modbus_servAddr.sin_family  = AF_INET;
    modbus_servAddr.sin_port    = htons(port);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if (inet_pton(AF_INET, ip_addr, &modbus_servAddr.sin_addr) <= 0) {

        printf("[ERROR] Failed to convert IP address! \n");
        return false;
    }

    if (connect(modbus_socket, (struct sockaddr *)&modbus_servAddr, sizeof(modbus_servAddr)) < 0) {

        printf("[ERROR] Failed to connect modbus! \n");
        return false;
    }

    is_modbus_connect = true;

    return is_modbus_connect;
}

void LK_UR5e::modbus_get_tcpPosition(UR_tcpPosition* tcp_position) {

    if (is_modbus_connect == false) {return;}

    UR_MODBUS_HEX_UNIT unit1(PM_THOUSAND);
    UR_MODBUS_HEX_UNIT unit2(PM_TEN);

    // tcp_x
    if (get_modbus((char*)&TCP_X, &tcp_position->X, unit2) == false) {return;}
    // tcp_y
    if (get_modbus((char*)&TCP_Y, &tcp_position->Y, unit2) == false) {return;}
    // tcp_z
    if (get_modbus((char*)&TCP_Z, &tcp_position->Z, unit2) == false) {return;}
    // tcp_rx
    if (get_modbus((char*)&TCP_RX, &tcp_position->RX, unit1) == false) {return;}
    // tcp_ry
    if (get_modbus((char*)&TCP_RY, &tcp_position->RY, unit1) == false) {return;}
    // tcp_rz
    if (get_modbus((char*)&TCP_RZ, &tcp_position->RZ, unit1) == false) {return;}
}

void LK_UR5e::modbus_get_jointPosition(UR_jointPosition* joint_position) {

    if (is_modbus_connect == false) {return;}

    UR_MODBUS_HEX_UNIT unit(PM_THOUSAND);

    // base
    if (get_modbus((char*)&BASE, &joint_position->base, unit) == false) {return;}
    // shoulder
    if (get_modbus((char*)&SHOULDER, &joint_position->shoulder, unit) == false) {return;}
    // elbow
    if (get_modbus((char*)&ELBOW, &joint_position->elbow, unit) == false) {return;}
    // wrist1
    if (get_modbus((char*)&WRIST1, &joint_position->wrist1, unit) == false) {return;}
    // wrist2
    if (get_modbus((char*)&WRIST2, &joint_position->wrist2, unit) == false) {return;}
    // wrist3
    if (get_modbus((char*)&WRIST3, &joint_position->wrist3, unit) == false) {return;}
}

void LK_UR5e::modbus_get_jointCurrent(UR_jointCurrent* joint_current) {

    if (is_modbus_connect == false) {return;}

    UR_MODBUS_HEX_UNIT unit(PM_THOUSAND);

    // base
    if (get_modbus((char*)&BASE_CURRENT, &joint_current->base, unit) == false) {return;}
    // shoulder
    if (get_modbus((char*)&SHOULDER_CURRENT, &joint_current->shoulder, unit) == false) {return;}
    // elbow
    if (get_modbus((char*)&ELBOW_CURRENT, &joint_current->elbow, unit) == false) {return;}
    // wrist1
    if (get_modbus((char*)&WRIST1_CURRENT, &joint_current->wrist1, unit) == false) {return;}
    // wrist2
    if (get_modbus((char*)&WRIST2_CURRENT, &joint_current->wrist2, unit) == false) {return;}
    // wrist3
    if (get_modbus((char*)&WRIST3_CURRENT, &joint_current->wrist3, unit) == false) {return;}
}

void LK_UR5e::modbus_close() {

    if (is_modbus_connect == false) {return;}
    close(modbus_socket);
}
// < /define : socket modbus communication >

// < define : modbus communication >
float LK_UR5e::modbus_hex2value_f(uint8_t* buf, UR_MODBUS_HEX_UNIT unit) {

    uint16_t row = 0;

    ((uint8_t*)&row)[0] = buf[10];
    ((uint8_t*)&row)[1] = buf[9];

    if (row > 32767 && (unit == PM_THOUSAND || unit == PM_TEN)) {

        row = 65535 - row;
        if(unit == PM_THOUSAND) {return (-1*(int)row)/1000.0;}
        if(unit==PM_TEN){return (-1*(int)row)/10.0;}
    }
    else {
        if(unit==THOUSAND||unit==PM_THOUSAND){return ((int)row)/1000.0;}
        if(unit==TEN||unit==PM_TEN){return ((int)row)/10.0;}
    }
}

double LK_UR5e::modbus_hex2value_d(uint8_t* buf, UR_MODBUS_HEX_UNIT unit) {

    uint16_t row = 0;

    ((uint8_t*)&row)[0]=buf[10];
    ((uint8_t*)&row)[1]=buf[9];
    
    if( row > 32767 && (unit==PM_THOUSAND || unit==PM_TEN))
    {
        row = 65535-row;
        if(unit==PM_THOUSAND){return (-1*(int)row)/1000.0;}
        if(unit==PM_TEN){return (-1*(int)row)/10.0;}
    }
    else
    {
        if(unit==THOUSAND||unit==PM_THOUSAND){return ((int)row)/1000.0;}
        if(unit==TEN||unit==PM_TEN){return ((int)row)/10.0;}
    }
}

bool LK_UR5e::get_modbus(const char* REGISTERS, float* getvalue, UR_MODBUS_HEX_UNIT type) {

    if (is_modbus_connect == false) {

        is_modbus_connect = false;
        modbus_close();
        printf("modbus close \n");
        return false;
    }

    send(this->modbus_socket, REGISTERS, 12, 0);
    modbus_valread = recv(this->modbus_socket, modbus_buffer, 1024, 0);

    if (modbus_valread == -1) {return false;}

    *getvalue = modbus_hex2value_d(modbus_buffer, type);

    return true;
}

bool LK_UR5e::get_modbus(const char* REGISTERS, double* getvalue, UR_MODBUS_HEX_UNIT type) {

    if (is_modbus_connect == false) {

        is_modbus_connect = false;
        modbus_close();
        printf("modbus close \n");
        return false;
    }

    send(this->modbus_socket, REGISTERS, 12, 0);
    modbus_valread = recv(this->modbus_socket, modbus_buffer, 1024, 0);

    if (modbus_valread == -1) {return false;}

    *getvalue = modbus_hex2value_d(modbus_buffer, type);

    return true;
}
// < /define : modbus communication >