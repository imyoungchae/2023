#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h> 

#include <list>     //####
#include <thread>   //####
#include <sstream>  //####
#include <iomanip>  //####


#include "lk_urStruct.hpp"
//#include "dk_time.hpp"
/*
writer: DaeKwan ko
date  : 20190728
e-mail: daekwanko123@gmail.com
*/
template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

//modbus REGISTERS ref: https://www.universal-robots.com/how-tos-and-faqs/how-to/ur-how-tos/modbus-server-16377/

//UR5e MODBUS-TCP Protocol  : V1_1
//MODBUS-TCP/IP ADU         : MBAPHeader + FunctionCode + Data
//      - MBAPHeader        : TRANSACTIONID(2byte) + PROTOCOLID(2byte) + UNITLENGTH(2byte) + UNITID(1byte)
//      - FunctionCode      : FUNCTIONCODE(1byte)
//      - Data              : ADDRESS(2byte) + Data(2byte)

#define TRANSACTIONID = "\x00\x04"//Master PC ID your choise 0x00XX
#define PROTOCOLID = "\x00\x00"//MODBUS-TCP ID
#define UNITLENGTH = "\x00\x06"
#define UNITID = "\x00"//UR5e ID: 00(=0x00)
#define FUNCTIONCODE_READ = "\x03"//MODBUS PROTOCOL
#define FUNCTIONCODE_WRITE = "\x06"//MODBUS PROTOCOL


#define joint_status_ALL "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x0e\x00\x1a"
//=========================== JointPosition =================================
#define joint_position_ALL "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x0e\x00\x06"
#define base "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x0e\x00\x01"
#define shoulder "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x0f\x00\x01"
#define elbow "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x10\x00\x01"
#define wrist1 "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x11\x00\x01"
#define wrist2 "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x12\x00\x01"
#define wrist3 "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x13\x00\x01"
//=========================== JointPosition =================================


//=========================== JointCurrent ==================================
#define current_ALL "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x22\x00\x06"
#define basecurrent "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x22\x00\x01"
#define shouldercurrent "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x23\x00\x01"
#define elbowcurrent "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x24\x00\x01"
#define wrist1current "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x25\x00\x01"
#define wrist2current "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x26\x00\x01"
#define wrist3current "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x27\x00\x01"
//=========================== JointCurrent ==================================

//=========================== TCP Position ==================================
#define tcp_position_ALL "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x90\x00\x06"
#define tcp_x "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x90\x00\x01"
#define tcp_y "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x91\x00\x01"
#define tcp_z "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x92\x00\x01"
#define tcp_rx "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x93\x00\x01"
#define tcp_ry "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x94\x00\x01"
#define tcp_rz "\x00\x04\x00\x00\x00\x06\x00\x03\x01\x95\x00\x01"
//=========================== TCP Position ==================================

#define DK_PI   3.1415926535897932384626433832795
#define DK_2PI  6.283185307179586476925286766559

#define DK_RAD2DEG   180.0/3.1415926535897932384626433832795
#define DK_DEG2RAD   3.1415926535897932384626433832795/180.0

/*enum UR_MODBUS_HEX_UNIT{THOUSAND,TEN,PM_THOUSAND,PM_TEN}; */

class DK_UR5e_Robot
{

private:
    // Ethernet socket
    int Ethernet_sock = 0;
    int Ethernet_valread;
    struct sockaddr_in Ethernet_serv_addr;
    char Ethernet_buffer[1024] = {0};
    bool is_Ethernet_connect =false;
    

// <Ethernet>
public:
    bool Ethernet_connect(const char* ip_add,int port);
    void Set_Ethernet_movej(UR_TCP_Position* tcp_position,float max_vel,float max_acc);
    void Set_Ethernet_movel(UR_TCP_Position* tcp_position,float max_vel,float max_acc);
    void Set_Ethernet_servoj(UR_TCP_Position* tcp_position,float timeout);
    void Set_Ethernet_profile_servoj(UR_TCP_Position& from_tcp_pos,UR_TCP_Position& to_tcp_pos,float max_v,float max_w);
    void Ethernet_close();
// </Ethernet>


// <Position Control>
// this is universal robot function in <Position Control> for realtime control base on https://github.com/ros-industrial/ur_modern_driver/blob/master/src/ur_driver.cpp
// 위 <Ethernet>는 거의 테스트용이며 1초정도 주기를 가지는 단순한 제어하는데 적합하다.
// 그러나 universal robot을 Position Control으로 제어하는데 문제점이 있는데 그것은 universal robot의 제어기 main process의 동작 clock을 알수 없어 synchronize문제가 발생한다.
// universal robot의 제어기는 synchronize문제 발생시 robot의 break를 동작시키기 때문에 robot의 상태가 불안정해진다. 
// 그래서 github의 ur_modern_driver의 아이디어는 robot의 제어기에서 PC의 server로 접속하여 PC의 clock에 의존적으로 만든다는 개념이다.
// PC의 clock은 알 수 있으므로 동기화 문제를 해결할 수 있다.
public:
    bool Upload_UR_Script(const char* myserver_ip_add,int myserver_port,const char* controler_ip_add,int controler_port);// Thread 기능이 있는 로봇 컨트롤러 스크립트 업로드
    bool UR_Server_Open(const char* ip_add,int port);// 내 PC 서버 열기
    bool UR_Server_Process();//this function call to thread.
    void Get_UR_Status(UR_TCP_Position* tcp_position,UR_Joint_Position* joint_array_rad,UR_Joint_Current* joint_array_A);
    void Set_UR_Control(UR_TCP_Position* tcp_position,int on_off);//on_off = 0 close Universal Robot Script, on_off = 1 run Universal Robot Script
    void Get_UR_Control(UR_TCP_Position* tcp_position,int* on_off);//on_off = 0 close Universal Robot Script, on_off = 1 run Universal Robot Script
    bool UR_Server_Close();// 내 PC 서버 닫기

private:
    bool is_Ethernet_Realtime_Server_open=false;
    UR_TCP_Position realtime_tcp_position;
    bool first_realtime_tcp_position_send = true;
    int UniversalRobot_sockfd_;
    int MyPC_Server_sockfd_;
    int keepalive=1;

    unsigned int bytes_written;
	int urserver_tmp;
	unsigned char urserver_buf[28];

// </Position Control>

// <calculation>
public:
    void forward(UR_Joint_Position* input_joint, transform_M* output_T);
    void transform2tcp_position(transform_M* intput_Transfom, UR_TCP_Position* output_position);
    void tcp_position2transform(UR_TCP_Position* input_position,transform_M* output_Transform);
    void tcp_position2RPY_position(UR_TCP_Position* input_position,UR_RPY_Position* output_Transform);
    void RPY_position2transform(UR_RPY_Position* input_position,transform_M* output_Transform);
    
    void Euler_position2transform(UR_Euler_Position* input_position,transform_M* output_Transform);

    void transform2RPY_position(transform_M* intput_Transform, UR_RPY_Position* output_position);
    bool tcp_position_boundry_check(UR_TCP_Position& ref_position, UR_TCP_Position& target_position,double pos_error,double ang_error);
    void Get_DH_T(transform_M* output_TM,double DH_theta,double DH_a,double DH_d,double DH_alpha);
    void matmul(double* frontT,double* backT,double* outT);
    void tcp_position_linear_add(UR_TCP_Position* ref_position, UR_TCP_Position* add_position, UR_TCP_Position* output_position);
    void tcp_position_linear_add(UR_RPY_Position* ref_position, UR_TCP_Position* add_position, UR_TCP_Position* output_position);
    void tcp_position_linear_add(UR_TCP_Position* ref_position, UR_RPY_Position* add_position, UR_TCP_Position* output_position);


    void tcp_position_linear_add_clip(UR_TCP_Position* ref_position, UR_TCP_Position* add_position, UR_TCP_Position* output_position,UR_Clip_RPY* clip_vector);
    void tcp_position_linear_add_clip(UR_RPY_Position* ref_position, UR_TCP_Position* add_position, UR_TCP_Position* output_position,UR_Clip_RPY* clip_vector);
    void tcp_position_linear_add_clip(UR_TCP_Position* ref_position, UR_RPY_Position* add_position, UR_TCP_Position* output_position,UR_Clip_RPY* clip_vector);


    void tcp_position_linear_subtract(UR_TCP_Position* ref_position, UR_TCP_Position* sub_position, UR_TCP_Position* output_position);
    void tcp_position_linear_subtract(UR_TCP_Position* ref_position, UR_RPY_Position* sub_position, UR_TCP_Position* output_position);
    void tcp_position_linear_subtract(UR_RPY_Position* ref_position, UR_RPY_Position* sub_position, UR_TCP_Position* output_position);
    void rpy_position_linear_subtract(UR_RPY_Position* ref_position, UR_RPY_Position* sub_position, UR_RPY_Position* output_position);

    void uint16todouble(uint8_t* low,uint8_t* high,double* out_data,double divide);
// </calculation>

// <socket modbus communication>
public:
    bool modbus_connect(const char* ip_add,int port);
    void Get_modbus_TCP_Position(UR_TCP_Position* tcp_position);
    void Get_modbus_TCP_Position_ALL(UR_TCP_Position* tcp_position);
    void Get_modbus_joint(UR_Joint_Position* joint_array_rad);
    void Get_modbus_joint_ALL(UR_Joint_Position* joint_array_rad);
    void Get_modbus_joint_status_ALL(UR_Joint_Position* joint_array_rad,UR_Joint_Current* joint_array_A);
    void Get_modbus_joint_current(UR_Joint_Current* joint_array_A);
    void Get_modbus_joint_current_ALL(UR_Joint_Current* joint_array_A);
    void modbus_close();
// </socket modbus communication>

// <modbus communication>
private:
    int modbus_socket = 1;
    int modbus_valread;
    struct sockaddr_in modbusserv_addr;
    bool modbus_is_connect =false;
    uint8_t modbus_buffer[1024]={0x00,};
    float modbus_hex_to_value_f(uint8_t* buf,UR_MODBUS_HEX_UNIT unit);
    double modbus_hex_to_value_d(uint8_t* buf,UR_MODBUS_HEX_UNIT unit);
    bool Get_modbus(const char* REGISTERS,double* getvalue,UR_MODBUS_HEX_UNIT type);
    bool Get_modbus(const char* REGISTERS,float* getvalue,UR_MODBUS_HEX_UNIT type);
// </modbus communication>

// <Robot info>
private:
    //DH-parameter: "introdution to robotics - SAEEDD B,NIKU" book
    //ur5e ref: https://www.universal-robots.com/how-tos-and-faqs/faq/ur-faq/parameters-for-calculations-of-kinematics-and-dynamics-45257/
    const double d1 =  0.1625*1000;//(Unit: mm)
    const double a2 = -0.425*1000;//(Unit: mm)
    const double a3 = -0.3922*1000;//(Unit: mm)
    const double d4 =  0.1333*1000;//(Unit: mm)
    const double d5 =  0.0997*1000;//(Unit: mm)
    const double d6 =  0.0996*1000;//(Unit: mm)
    // RobotControl state
    const int communication_wait_time = 2;// (Unit: ms)
    const double ControlerBox_version = 5.4;
    const unsigned int MULT_JOINTSTATE_ = 1000000;
// <Robot info>

// <Test>
public:
    //apply function
    bool Set_movel_wait(UR_TCP_Position* tcp_position,float max_vel,float max_acc,int timeout);
    bool trajectory_movel(std::list<UR_TCP_Position>::iterator start, std::list<UR_TCP_Position>::iterator end,float maxval,int wait_time);


private:
    UR_TCP_Position get_modbus_tcp_position;
    UR_Joint_Position get_modbus_joint_position;
    UR_Joint_Current get_modbus_joint_current;
// </Test>

//*/sigleton
public:
    virtual ~DK_UR5e_Robot(){instance_flag=false; modbus_close();Ethernet_close();};
    static DK_UR5e_Robot* get_instance();
    static bool thread_flag;

private:
    DK_UR5e_Robot(){};
    static bool instance_flag;
    static DK_UR5e_Robot* instance;
//*/    
};

DK_UR5e_Robot* DK_UR5e_Robot::instance=NULL;
bool DK_UR5e_Robot::instance_flag=false;


DK_UR5e_Robot* DK_UR5e_Robot::get_instance()
{
    if(instance_flag==false)
    {
        instance= new DK_UR5e_Robot();
        instance_flag=true;
    }
    return instance;
}

bool DK_UR5e_Robot::UR_Server_Process()
{
    
    if (is_Ethernet_Realtime_Server_open == false) {
		printf("UrDriver::servoj called without a reverse connection present. Keepalive: %d\n",0);
		return false;
	}

    Get_modbus_TCP_Position_ALL(&get_modbus_tcp_position);    
    // Get_modbus_joint_ALL(&get_modbus_joint_position);
    // Get_modbus_joint_current_ALL(&get_modbus_joint_current);

	for (int i = 0; i < 6; i++) 
    {
        if(i<3)
        {
            urserver_tmp = htonl((int)(realtime_tcp_position.element[i] * (MULT_JOINTSTATE_ /1000.0) ));
        }else
        {
            urserver_tmp = htonl((int)(realtime_tcp_position.element[i] * MULT_JOINTSTATE_));
        }
        urserver_buf[i * 4] = urserver_tmp & 0xff;
		urserver_buf[i * 4 + 1] = (urserver_tmp >> 8) & 0xff;
		urserver_buf[i * 4 + 2] = (urserver_tmp >> 16) & 0xff;
		urserver_buf[i * 4 + 3] = (urserver_tmp >> 24) & 0xff;
	}
    urserver_tmp = htonl((int) keepalive);
	urserver_buf[6 * 4] = urserver_tmp & 0xff;
	urserver_buf[6 * 4 + 1] = (urserver_tmp >> 8) & 0xff;
	urserver_buf[6 * 4 + 2] = (urserver_tmp >> 16) & 0xff;
	urserver_buf[6 * 4 + 3] = (urserver_tmp >> 24) & 0xff;

	bytes_written = write(UniversalRobot_sockfd_,urserver_buf, 28);
    
    return true;
}


void DK_UR5e_Robot::Get_UR_Status(UR_TCP_Position* tcp_position=nullptr,UR_Joint_Position* joint_array_rad=nullptr,UR_Joint_Current* joint_array_A=nullptr)
{
    if(tcp_position!=nullptr)
    {
        *tcp_position = get_modbus_tcp_position;
    }
    if(joint_array_rad!=nullptr)
    {
        *joint_array_rad = get_modbus_joint_position;
    }
    if(joint_array_A!=nullptr)
    {
        *joint_array_A = get_modbus_joint_current;
    }
}

bool DK_UR5e_Robot::UR_Server_Open(const char* ip_add,int port)
{

    if(is_Ethernet_Realtime_Server_open==true){return false;}
    char buffer[256];
	struct sockaddr_in serv_addr;
    struct sockaddr_in cli_addr;
	socklen_t clilen;

	int n;
    int flag;

	UniversalRobot_sockfd_ = -1;
	MyPC_Server_sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
	
    if (MyPC_Server_sockfd_ < 0)
    {
		printf("ERROR opening socket for reverse communication\n");
        return false;
	}

	bzero((u_char *) &serv_addr, sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = inet_addr(ip_add);
	serv_addr.sin_port = htons(port);
	flag = 1;

	setsockopt(MyPC_Server_sockfd_, IPPROTO_TCP, TCP_NODELAY,(char *) &flag, sizeof(int));
    setsockopt(MyPC_Server_sockfd_, IPPROTO_TCP, TCP_QUICKACK,(char *) &flag, sizeof(int));
	setsockopt(MyPC_Server_sockfd_, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(int));

	if (bind(MyPC_Server_sockfd_, (struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
    {
		printf("%s : %d ERROR on binding socket for reverse communication\n",ip_add,port);
        
        return false;
	}
	listen(MyPC_Server_sockfd_, 5);


	clilen = sizeof(cli_addr);
	UniversalRobot_sockfd_ = accept(MyPC_Server_sockfd_, (struct sockaddr *) &cli_addr, &clilen);
	
    if (UniversalRobot_sockfd_ < 0) 
    {
		printf("ERROR on accepting reverse communication\n");
		return false;
	}

    printf(" UR robot Successfully Connect at MyPC \n");
    is_Ethernet_Realtime_Server_open=true;
	return is_Ethernet_Realtime_Server_open;
}

bool DK_UR5e_Robot::UR_Server_Close()
{
    if(is_Ethernet_Realtime_Server_open==false){return false;}
	Set_UR_Control(&realtime_tcp_position, 0);
    UR_Server_Process();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
	is_Ethernet_Realtime_Server_open = false;
	close(UniversalRobot_sockfd_);
    close(MyPC_Server_sockfd_);
    return true;
}

void DK_UR5e_Robot::Set_UR_Control(UR_TCP_Position* tcp_position,int on_off)
{
    if(is_Ethernet_Realtime_Server_open==false){return;}

    realtime_tcp_position=(*tcp_position);
    keepalive = on_off;
}

void DK_UR5e_Robot::Get_UR_Control(UR_TCP_Position* tcp_position,int* on_off)
{
    if(is_Ethernet_Realtime_Server_open==false){return;}

    (*tcp_position)=realtime_tcp_position;
    (*on_off) = keepalive;
}

bool DK_UR5e_Robot::Upload_UR_Script(const char* myserver_ip_add,int myserver_port,const char* controler_ip_add,int controler_port)
{
    if(is_Ethernet_connect==false){Ethernet_connect(controler_ip_add,controler_port);}
    if(is_Ethernet_connect==false){printf("not connected Universal Robot Ethernet\n");return false;}
    if(modbus_is_connect==false){modbus_connect(controler_ip_add,502);}
    if(modbus_is_connect==false){printf("not connectde Universal Robot MODBUS\n");return false;}

    float servoj_time_ = 0.008f;//time [S], range [0.002~infinit] 
    float servoj_lookahead_time_= 0.1f;//time [S], range [0.03,0.2] smoothens the trajectory with this lookahead time
    float servoj_gain_=300.0f;//

    //<Universal Robot Script>
    std::string cmd_str;
    char buf[128];
    
	cmd_str = "def driverProg():\n";

	sprintf(buf, "\tMULT_jointstate = %i\n", MULT_JOINTSTATE_);
	cmd_str += buf;
	cmd_str += "\tSERVO_IDLE = 0\n";
	cmd_str += "\tSERVO_RUNNING = 1\n";
	cmd_str += "\tcmd_servo_state = SERVO_IDLE\n";
    cmd_str += "\tcmd_servo_q = get_actual_joint_positions()\n";
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

	if(ControlerBox_version >= 3.1)
    {
        sprintf(buf, "\t\t\t\tservoj(q, t=%.4f, lookahead_time=%.4f, gain=%.0f)\n",
				servoj_time_, servoj_lookahead_time_, servoj_gain_);
    }else
    {
        sprintf(buf, "\t\t\t\tservoj(q, t=%.4f)\n", servoj_time_);
    }

	cmd_str += buf;

	cmd_str += "\t\t\telse:\n";
	cmd_str += "\t\t\t\tsync()\n";
	cmd_str += "\t\t\tend\n";
	cmd_str += "\t\tend\n";
	cmd_str += "\tend\n";

	sprintf(buf, "\tsocket_open(\"%s\", %i)\n", myserver_ip_add,
			myserver_port);
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
    //</Universal Robot Script>
    char const *pchar = cmd_str.c_str();

    send(Ethernet_sock,pchar,cmd_str.length(),0);
    usleep(100);
    Get_modbus_TCP_Position_ALL(&realtime_tcp_position);
    printf("Successfully Upload Universal Robot Script\n");
    return true;
}


bool DK_UR5e_Robot::Set_movel_wait(UR_TCP_Position* tcp_position,float max_vel,float max_acc,int timeout)
{
    bool is_successful = false;

    if(modbus_is_connect==false || is_Ethernet_connect==false || is_Ethernet_Realtime_Server_open == true){return is_successful;}

    int timeout_count = (timeout/communication_wait_time);
    int now_count=0;
    Set_Ethernet_movel(tcp_position,max_vel,max_acc);
    std::this_thread::sleep_for(std::chrono::milliseconds(communication_wait_time));

    while(!tcp_position_boundry_check(get_modbus_tcp_position,*tcp_position,0.5,0.03))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(communication_wait_time));

        now_count++;
        if(now_count>timeout_count)
        {
            printf("timeout \n");
            return is_successful;
        }
    }

    is_successful=true;
    return is_successful;

}

bool DK_UR5e_Robot::trajectory_movel(std::list<UR_TCP_Position>::iterator start, std::list<UR_TCP_Position>::iterator end,float maxval=0.3,int wait_time=1000)
{
    bool is_successful = false;

    if(modbus_is_connect==false || is_Ethernet_connect==false || is_Ethernet_Realtime_Server_open == true){return is_successful;}

    while(start!=end)
    {
        if(Set_movel_wait(&(*start),maxval,0.8,10000)==false)
        {
            return is_successful;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
        start++;
    }

    is_successful=true;
    return is_successful;
}


void DK_UR5e_Robot::Get_DH_T(transform_M* output_TM,double DH_theta,double DH_a,double DH_d,double DH_alpha)
{
    output_TM->r11=cos(DH_theta);
    output_TM->r12=-sin(DH_theta)*cos(DH_alpha);
    output_TM->r13=sin(DH_theta)*sin(DH_alpha);
    output_TM->x=DH_a*cos(DH_theta);
    
    output_TM->r21=sin(DH_theta);
    output_TM->r22=cos(DH_theta)*cos(DH_alpha);
    output_TM->r23=-cos(DH_theta)*sin(DH_alpha);
    output_TM->y=DH_a*sin(DH_theta);

    output_TM->r31=0;
    output_TM->r32=sin(DH_alpha);
    output_TM->r33=cos(DH_alpha);
    output_TM->z=DH_d;

    output_TM->wn=0;
    output_TM->wo=0;
    output_TM->wa=0;
    output_TM->w=1;
}

void DK_UR5e_Robot::matmul(double* frontT,double* backT,double* outT)
{
    for(int c =0;c<4;c++){for(int r =0;r<4;r++){outT[c+(r*4)]=0;for(int e=0;e<4;e++){outT[c+(r*4)]+=frontT[e+(r*4)]*backT[(e*4)+c];}}}
}



void DK_UR5e_Robot::transform2tcp_position(transform_M* intput_Transfom, UR_TCP_Position* output_position)
{
    //reference: "introduction to robotics - John j.Craig" book 73p
    //Converntion Equivalent Angle-axis 

    output_position->X=intput_Transfom->x;
    output_position->Y=intput_Transfom->y;
    output_position->Z=intput_Transfom->z;

    double theta = acos( (intput_Transfom->r11+intput_Transfom->r22+intput_Transfom->r33-1) /2.0);
   
    if( (theta<0.001 && theta>-0.001))//robot unstable exception 
    {
        output_position->RX=0.0;
        output_position->RY=0.0;
        output_position->RZ=0.0;
        // printf("theta==0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // printf("theta==0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // printf("theta==0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // printf("theta==0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // printf("theta==0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // printf("theta==0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        return;
    }

    if( (theta<(DK_PI+0.1) && theta>(DK_PI-0.1)))//robot unstable exception 
    {
        // printf("theta==PI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // printf("theta==PI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // printf("theta==PI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // printf("theta==PI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // printf("theta==PI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // printf("theta==PI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }


    double multi = (1.0/(2*sin(theta)));

    double RX = multi*(intput_Transfom->r32-intput_Transfom->r23);
    double RY = multi*(intput_Transfom->r13-intput_Transfom->r31);
    double RZ = multi*(intput_Transfom->r21-intput_Transfom->r12);

    double vector_norm = sqrt( (RX*RX) + (RY*RY) + (RZ*RZ));
    // 이건 책에 없다....
    // 이론상 [RX, RY, RZ]는 회전 벡터이므로 크기가 1이다.
    // 그러나 소수점에서 계산상의 손실로 [RX, RY, RZ]가 1을 보장할 수 없다. 
    // 때문에 회전 벡터의 크기인 vector_norm의 값은 1이 나오도록 맞추는 작업이 필요함... 미세한 안전성 향상 효과 .......
    // 위 아래 작업은 이를 위한 것....

    output_position->RX = theta*(RX/vector_norm);
    output_position->RY = theta*(RY/vector_norm);
    output_position->RZ = theta*(RZ/vector_norm);
}

void DK_UR5e_Robot::tcp_position2transform(UR_TCP_Position* input_position,transform_M* output_Transform)
{
    //reference: "introduction to robotics - John j.Craig" book 73p
    //Converntion Equivalent Angle-axis 

    output_Transform->x=input_position->X;
    output_Transform->y=input_position->Y;
    output_Transform->z=input_position->Z;
    output_Transform->w=1;
    output_Transform->wn=0;
    output_Transform->wo=0;
    output_Transform->wa=0;
    
    
    double kx=input_position->RX;
    double ky=input_position->RY;
    double kz=input_position->RZ;

    double theta= sqrt((kx*kx)+(ky*ky)+(kz*kz));
    
    if( (theta<0.00001 && theta>-0.00001))//robot unstable exception 
    {
        output_Transform->r11=1.0;
        output_Transform->r12=0.0;
        output_Transform->r13=0.0;
        output_Transform->r21=0.0;
        output_Transform->r22=1.0;
        output_Transform->r23=0.0;
        output_Transform->r31=0.0;
        output_Transform->r32=0.0;
        output_Transform->r33=1.0;
        return;
    }

    kx=kx/theta;
    ky=ky/theta;
    kz=kz/theta;
 
    double ctheta= cos(theta);
    double stheta= sin(theta);
    double vtheta= 1.0-cos(theta);

    output_Transform->r11=(kx*kx*vtheta) + (ctheta);
    output_Transform->r12=(kx*ky*vtheta) - (kz*stheta);
    output_Transform->r13=(kx*kz*vtheta) + (ky*stheta);

    output_Transform->r21=(ky*kx*vtheta) + (kz*stheta);
    output_Transform->r22=(ky*ky*vtheta) + (ctheta);
    output_Transform->r23=(ky*kz*vtheta) - (kx*stheta);

    output_Transform->r31=(kx*kz*vtheta) - (ky*stheta);
    output_Transform->r32=(ky*kz*vtheta) + (kx*stheta);
    output_Transform->r33=(kz*kz*vtheta) + (ctheta);
}


void DK_UR5e_Robot::tcp_position2RPY_position(UR_TCP_Position* input_position,UR_RPY_Position* output_Transform)
{
    transform_M temp;
    tcp_position2transform(input_position,&temp);
    transform2RPY_position(&temp,output_Transform);
}



void DK_UR5e_Robot::RPY_position2transform(UR_RPY_Position* input_position,transform_M* output_Transform)
{
    //reference: "introdution to robotics - SAEEDD B,NIKU" book 67p
    output_Transform->x=input_position->X;
    output_Transform->y=input_position->Y;
    output_Transform->z=input_position->Z;

    double pi_a = input_position->Roll;
    double pi_o = input_position->Pitch;
    double pi_n = input_position->Yaw;
    
    output_Transform->r11=cos(pi_a)*cos(pi_o);
    output_Transform->r12=(cos(pi_a)*sin(pi_o)*sin(pi_n))-(sin(pi_a)*cos(pi_n));
    output_Transform->r13=(cos(pi_a)*sin(pi_o)*cos(pi_n))+(sin(pi_a)*sin(pi_n));

    output_Transform->r21=sin(pi_a)*cos(pi_o);
    output_Transform->r22=(sin(pi_a)*sin(pi_o)*sin(pi_n))+(cos(pi_a)*cos(pi_n));
    output_Transform->r23=(sin(pi_a)*sin(pi_o)*cos(pi_n))-(cos(pi_a)*sin(pi_n));

    output_Transform->r31=-sin(pi_o);
    output_Transform->r32=cos(pi_o)*sin(pi_n);
    output_Transform->r33=cos(pi_o)*cos(pi_n);
    
    output_Transform->wn=0;
    output_Transform->wo=0;
    output_Transform->wa=0;
    output_Transform->w=1;
}


void DK_UR5e_Robot::Euler_position2transform(UR_Euler_Position* input_position,transform_M* output_Transform)
{
    //reference: "introdution to robotics - SAEEDD B,NIKU" book 70p
    output_Transform->x=input_position->X;
    output_Transform->y=input_position->Y;
    output_Transform->z=input_position->Z;

    EulerAngles angle = ToEulerAngles(ToQuaternion(input_position->Yaw,input_position->Pitch,input_position->Roll));
    
    double pi_a = angle.roll;
    double theta_o = angle.pitch;
    double phi_a = angle.yaw;
    
    output_Transform->r11=(cos(pi_a)*cos(theta_o)*cos(phi_a))+(-sin(pi_a)*sin(phi_a));
    output_Transform->r12=(-cos(pi_a)*cos(theta_o)*sin(phi_a))+(-sin(pi_a)*cos(phi_a));
    output_Transform->r13=cos(pi_a)*sin(theta_o);

    output_Transform->r21=(sin(pi_a)*cos(theta_o)*cos(phi_a))+(cos(pi_a)*sin(phi_a));
    output_Transform->r22=(-sin(pi_a)*cos(theta_o)*sin(phi_a))+(cos(pi_a)*cos(phi_a));
    output_Transform->r23=sin(pi_a)*sin(theta_o);

    output_Transform->r31=-sin(theta_o)*cos(phi_a);
    output_Transform->r32=sin(theta_o)*sin(phi_a);
    output_Transform->r33=cos(theta_o);
    
    output_Transform->wn=0;
    output_Transform->wo=0;
    output_Transform->wa=0;
    output_Transform->w=1;
}

void DK_UR5e_Robot::transform2RPY_position(transform_M* intput_Transform, UR_RPY_Position* output_position)
{
    output_position->X=intput_Transform->x;
    output_position->Y=intput_Transform->y;
    output_position->Z=intput_Transform->z;
   
    output_position->Roll = atan2(  intput_Transform->ny,
                                    intput_Transform->nx);

    output_position->Pitch = atan2(  -intput_Transform->nz,
                                    ( (intput_Transform->nx*cos(output_position->Roll)) + (intput_Transform->ny*sin(output_position->Roll)) ) );

    output_position->Yaw = atan2(   ( -(intput_Transform->ay *cos(output_position->Roll)) + (intput_Transform->ax * sin(output_position->Roll)) ),
                                    (  (intput_Transform->oy*cos(output_position->Roll))  - (intput_Transform->ox*sin(output_position->Roll))   ) );

}

bool DK_UR5e_Robot::tcp_position_boundry_check(UR_TCP_Position& ref_position, UR_TCP_Position& target_position,double pos_error=0.05,double ang_error=0.05)
{
    return (ref_position.X>(target_position.X-pos_error) && ref_position.X<(target_position.X+pos_error))
            && (ref_position.Y>(target_position.Y-pos_error) && ref_position.Y<(target_position.Y+pos_error))
            && (ref_position.Z>(target_position.Z-pos_error) && ref_position.Z<(target_position.Z+pos_error))
            && (ref_position.RX>(target_position.RX-ang_error) && ref_position.RX<(target_position.RX+ang_error))
            && (ref_position.RY>(target_position.RY-ang_error) && ref_position.RY<(target_position.RY+ang_error))
            && (ref_position.RZ>(target_position.RZ-ang_error) && ref_position.RZ<(target_position.RZ+ang_error));
            
}


void DK_UR5e_Robot::forward(UR_Joint_Position* input_joint, transform_M* output_T)
{
    transform_M T1; Get_DH_T(&T1,input_joint->base_rad,0,d1,DK_PI/2);
    transform_M T2; Get_DH_T(&T2,input_joint->shoulder_rad,a2,0,0);
    transform_M T3; Get_DH_T(&T3,input_joint->elbow_rad,a3,0,0);
    transform_M T4; Get_DH_T(&T4,input_joint->wrist1_rad,0,d4,DK_PI/2);
    transform_M T5; Get_DH_T(&T5,input_joint->wrist2_rad,0,d5,-DK_PI/2);
    transform_M T6; Get_DH_T(&T6,input_joint->wrist3_rad,0,d6,0);

    (*output_T)=T1*T2*T3*T4*T5*T6;
}
    
void DK_UR5e_Robot::modbus_close()
{
    if(modbus_is_connect==false){return;}
    close(modbus_socket);
}

void DK_UR5e_Robot::Ethernet_close()
{
    if(is_Ethernet_connect==false){return;}
    close(Ethernet_sock);
}




bool DK_UR5e_Robot::modbus_connect(const char* ip_add=NULL,int port=0)
{

    if(ip_add==NULL)
    {
        ip_add="127.0.0.1";
    }
    if(port==0)
    {
        port=502;
    }

    if ((modbus_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return false;
    }
   
    modbusserv_addr.sin_family = AF_INET;
    modbusserv_addr.sin_port = htons(port);

    int flag = 1;
	setsockopt(modbus_socket, IPPROTO_TCP, TCP_NODELAY,(u_char *) &flag, sizeof(int));
	setsockopt(modbus_socket, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(int));
 
       
    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, ip_add, &modbusserv_addr.sin_addr)<=0) 
    {
        printf("\nInvalid address/ Address not supported \n");
        return false;
    }
   
    if(connect(modbus_socket, (struct sockaddr *)&modbusserv_addr, sizeof(modbusserv_addr)) < 0 )
    {
        printf("\nConnection Failed \n");
        return false;
    }

    modbus_is_connect = true;
    return modbus_is_connect;

}
bool DK_UR5e_Robot::Ethernet_connect(const char* ip_add=NULL,int port=0)
{
    if(ip_add==NULL)
    {
        ip_add="127.0.0.1";
    }

    if(port==0)
    {
        port=30003;
    }
        
    if ((Ethernet_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return false;
    }

    int flag = 1;
	setsockopt(Ethernet_sock, IPPROTO_TCP, TCP_NODELAY,(u_char *) &flag, sizeof(int));
	setsockopt(Ethernet_sock, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(int));
    
    Ethernet_serv_addr.sin_family = AF_INET;
    Ethernet_serv_addr.sin_port = htons(port);
       
    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET,ip_add, &Ethernet_serv_addr.sin_addr)<=0) 
    {
        printf("\nInvalid address/ Address not supported \n");
        return false;
    }
   
    if(connect(Ethernet_sock, (struct sockaddr *)&Ethernet_serv_addr, sizeof(Ethernet_serv_addr)) < 0 )
    {
        printf("\nConnection Failed \n");
        return false;
    }

    is_Ethernet_connect = true;
    return is_Ethernet_connect;
}

void DK_UR5e_Robot::tcp_position_linear_add(UR_RPY_Position* ref_position, UR_TCP_Position* add_position, UR_TCP_Position* output_position)
{
    UR_RPY_Position ref_rpy_pos;
    UR_RPY_Position add_rpy_pos;
    UR_RPY_Position sub_rpy_pos;
    UR_RPY_Position output_rpy_pos;
    transform_M temp;

    output_rpy_pos.X = ref_position->X + add_position->X;
    output_rpy_pos.Y = ref_position->Y + add_position->Y;
    output_rpy_pos.Z = ref_position->Z + add_position->Z;

    
    ref_rpy_pos=*ref_position;
    tcp_position2transform(add_position,&temp);
    transform2RPY_position(&temp,&add_rpy_pos);

    output_rpy_pos.Yaw = ref_rpy_pos.Yaw + add_rpy_pos.Yaw;
    output_rpy_pos.Pitch = ref_rpy_pos.Pitch + add_rpy_pos.Pitch;
    output_rpy_pos.Roll = ref_rpy_pos.Roll + add_rpy_pos.Roll;

     if(output_rpy_pos.Yaw<-DK_PI)//만약 회전변화가 180deg 이상이면 => 로봇은 endpoint에 장착된 tool같은것 때문에 제약이 있을 수 있기때문에...
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw+DK_2PI);//위상 반전
    }
    else if (output_rpy_pos.Yaw>DK_PI)
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw-DK_2PI);
    }

    if(output_rpy_pos.Pitch<-DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch+DK_2PI);
    }
    else if (output_rpy_pos.Pitch>DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch-DK_2PI);
    }

    if(output_rpy_pos.Roll<-DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll+DK_2PI);
    }
    else if (output_rpy_pos.Roll>DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll-DK_2PI);
    }



    RPY_position2transform(&output_rpy_pos,&temp);
    transform2tcp_position(&temp,output_position);
}
void DK_UR5e_Robot::tcp_position_linear_add(UR_TCP_Position* ref_position, UR_RPY_Position* add_position, UR_TCP_Position* output_position){
    UR_RPY_Position ref_rpy_pos;
    UR_RPY_Position add_rpy_pos;
    UR_RPY_Position sub_rpy_pos;
    UR_RPY_Position output_rpy_pos;
    transform_M temp;
    output_rpy_pos.X = ref_position->X + add_position->X;
    output_rpy_pos.Y = ref_position->Y + add_position->Y;
    output_rpy_pos.Z = ref_position->Z + add_position->Z;

    tcp_position2transform(ref_position,&temp);
    transform2RPY_position(&temp,&ref_rpy_pos);

    add_rpy_pos=*add_position;

    output_rpy_pos.Yaw = ref_rpy_pos.Yaw + add_rpy_pos.Yaw;
    output_rpy_pos.Pitch = ref_rpy_pos.Pitch + add_rpy_pos.Pitch;
    output_rpy_pos.Roll = ref_rpy_pos.Roll + add_rpy_pos.Roll;

     if(output_rpy_pos.Yaw<-DK_PI)//만약 회전변화가 180deg 이상이면 => 로봇은 endpoint에 장착된 tool같은것 때문에 제약이 있을 수 있기때문에...
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw+DK_2PI);//위상 반전
    }
    else if (output_rpy_pos.Yaw>DK_PI)
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw-DK_2PI);
    }

    if(output_rpy_pos.Pitch<-DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch+DK_2PI);
    }
    else if (output_rpy_pos.Pitch>DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch-DK_2PI);
    }

    if(output_rpy_pos.Roll<-DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll+DK_2PI);
    }
    else if (output_rpy_pos.Roll>DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll-DK_2PI);
    }



    RPY_position2transform(&output_rpy_pos,&temp);
    transform2tcp_position(&temp,output_position);
}
void DK_UR5e_Robot::tcp_position_linear_add(UR_TCP_Position* ref_position, UR_TCP_Position* add_position, UR_TCP_Position* output_position)
{
    UR_RPY_Position ref_rpy_pos;
    UR_RPY_Position add_rpy_pos;
    UR_RPY_Position sub_rpy_pos;
    UR_RPY_Position output_rpy_pos;
    transform_M temp;
    output_rpy_pos.X = ref_position->X + add_position->X;
    output_rpy_pos.Y = ref_position->Y + add_position->Y;
    output_rpy_pos.Z = ref_position->Z + add_position->Z;

    tcp_position2transform(ref_position,&temp);
    transform2RPY_position(&temp,&ref_rpy_pos);

    tcp_position2transform(add_position,&temp);
    transform2RPY_position(&temp,&add_rpy_pos);

    output_rpy_pos.Yaw = ref_rpy_pos.Yaw + add_rpy_pos.Yaw;
    output_rpy_pos.Pitch = ref_rpy_pos.Pitch + add_rpy_pos.Pitch;
    output_rpy_pos.Roll = ref_rpy_pos.Roll + add_rpy_pos.Roll;

     if(output_rpy_pos.Yaw<-DK_PI)//만약 회전변화가 180deg 이상이면 => 로봇은 endpoint에 장착된 tool같은것 때문에 제약이 있을 수 있기때문에...
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw+DK_2PI);//위상 반전
    }
    else if (output_rpy_pos.Yaw>DK_PI)
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw-DK_2PI);
    }

    if(output_rpy_pos.Pitch<-DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch+DK_2PI);
    }
    else if (output_rpy_pos.Pitch>DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch-DK_2PI);
    }

    if(output_rpy_pos.Roll<-DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll+DK_2PI);
    }
    else if (output_rpy_pos.Roll>DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll-DK_2PI);
    }



    RPY_position2transform(&output_rpy_pos,&temp);
    transform2tcp_position(&temp,output_position);
}

void DK_UR5e_Robot::tcp_position_linear_add_clip(UR_RPY_Position* ref_position, UR_TCP_Position* add_position, UR_TCP_Position* output_position,UR_Clip_RPY* clip_vector)
{
    UR_RPY_Position ref_rpy_pos;
    UR_RPY_Position add_rpy_pos;
    UR_RPY_Position output_rpy_pos;
    transform_M temp;
    output_rpy_pos.X = ref_position->X + add_position->X;
    output_rpy_pos.Y = ref_position->Y + add_position->Y;
    output_rpy_pos.Z = ref_position->Z + add_position->Z;

    ref_rpy_pos=*ref_position;

    tcp_position2transform(add_position,&temp);
    transform2RPY_position(&temp,&add_rpy_pos);

    output_rpy_pos.Yaw = ref_rpy_pos.Yaw + add_rpy_pos.Yaw;
    output_rpy_pos.Pitch = ref_rpy_pos.Pitch + add_rpy_pos.Pitch;
    output_rpy_pos.Roll = ref_rpy_pos.Roll + add_rpy_pos.Roll;

    //printf("out X: %lf,Y: %lf,Z: %lf \n",output_rpy_pos.X,output_rpy_pos.Y,output_rpy_pos.Z);

    output_rpy_pos.X = clip(output_rpy_pos.X,clip_vector->X_Min,clip_vector->X_Max);
    output_rpy_pos.Y = clip(output_rpy_pos.Y,clip_vector->Y_Min,clip_vector->Y_Max);
    output_rpy_pos.Z = clip(output_rpy_pos.Z,clip_vector->Z_Min,clip_vector->Z_Max);

    output_rpy_pos.Pitch = clip(output_rpy_pos.Pitch,clip_vector->Pitch_Min,clip_vector->Pitch_Max);
    output_rpy_pos.Roll = clip(output_rpy_pos.Roll,clip_vector->Roll_Min,clip_vector->Roll_Max);
    output_rpy_pos.Yaw = clip(output_rpy_pos.Yaw,clip_vector->Yaw_Min,clip_vector->Yaw_Max);

     if(output_rpy_pos.Yaw<-DK_PI)//만약 회전변화가 180deg 이상이면 => 로봇은 endpoint에 장착된 tool같은것 때문에 제약이 있을 수 있기때문에...
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw+DK_2PI);//위상 반전
    }
    else if (output_rpy_pos.Yaw>DK_PI)
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw-DK_2PI);
    }

    if(output_rpy_pos.Pitch<-DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch+DK_2PI);
    }
    else if (output_rpy_pos.Pitch>DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch-DK_2PI);
    }

    if(output_rpy_pos.Roll<-DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll+DK_2PI);
    }
    else if (output_rpy_pos.Roll>DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll-DK_2PI);
    }

    
    RPY_position2transform(&output_rpy_pos,&temp);
    transform2tcp_position(&temp,output_position);
}
void DK_UR5e_Robot::tcp_position_linear_add_clip(UR_TCP_Position* ref_position, UR_RPY_Position* add_position, UR_TCP_Position* output_position,UR_Clip_RPY* clip_vector)
{
    UR_RPY_Position ref_rpy_pos;
    UR_RPY_Position add_rpy_pos;
    UR_RPY_Position output_rpy_pos;
    transform_M temp;
    output_rpy_pos.X = ref_position->X + add_position->X;
    output_rpy_pos.Y = ref_position->Y + add_position->Y;
    output_rpy_pos.Z = ref_position->Z + add_position->Z;

    tcp_position2transform(ref_position,&temp);
    transform2RPY_position(&temp,&ref_rpy_pos);

    add_rpy_pos=(*add_position);

    output_rpy_pos.Yaw = ref_rpy_pos.Yaw + add_rpy_pos.Yaw;
    output_rpy_pos.Pitch = ref_rpy_pos.Pitch + add_rpy_pos.Pitch;
    output_rpy_pos.Roll = ref_rpy_pos.Roll + add_rpy_pos.Roll;

    //printf("out X: %lf,Y: %lf,Z: %lf \n",output_rpy_pos.X,output_rpy_pos.Y,output_rpy_pos.Z);

    output_rpy_pos.X = clip(output_rpy_pos.X,clip_vector->X_Min,clip_vector->X_Max);
    output_rpy_pos.Y = clip(output_rpy_pos.Y,clip_vector->Y_Min,clip_vector->Y_Max);
    output_rpy_pos.Z = clip(output_rpy_pos.Z,clip_vector->Z_Min,clip_vector->Z_Max);

    output_rpy_pos.Pitch = clip(output_rpy_pos.Pitch,clip_vector->Pitch_Min,clip_vector->Pitch_Max);
    output_rpy_pos.Roll = clip(output_rpy_pos.Roll,clip_vector->Roll_Min,clip_vector->Roll_Max);
    output_rpy_pos.Yaw = clip(output_rpy_pos.Yaw,clip_vector->Yaw_Min,clip_vector->Yaw_Max);

     if(output_rpy_pos.Yaw<-DK_PI)//만약 회전변화가 180deg 이상이면 => 로봇은 endpoint에 장착된 tool같은것 때문에 제약이 있을 수 있기때문에...
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw+DK_2PI);//위상 반전
    }
    else if (output_rpy_pos.Yaw>DK_PI)
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw-DK_2PI);
    }

    if(output_rpy_pos.Pitch<-DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch+DK_2PI);
    }
    else if (output_rpy_pos.Pitch>DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch-DK_2PI);
    }

    if(output_rpy_pos.Roll<-DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll+DK_2PI);
    }
    else if (output_rpy_pos.Roll>DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll-DK_2PI);
    }
    
    RPY_position2transform(&output_rpy_pos,&temp);
    transform2tcp_position(&temp,output_position);
}
void DK_UR5e_Robot::tcp_position_linear_add_clip(UR_TCP_Position* ref_position, UR_TCP_Position* add_position, UR_TCP_Position* output_position,UR_Clip_RPY* clip_vector)
{
    UR_RPY_Position ref_rpy_pos;
    UR_RPY_Position add_rpy_pos;
    UR_RPY_Position sub_rpy_pos;
    UR_RPY_Position output_rpy_pos;
    transform_M temp;
    output_rpy_pos.X = ref_position->X + add_position->X;
    output_rpy_pos.Y = ref_position->Y + add_position->Y;
    output_rpy_pos.Z = ref_position->Z + add_position->Z;

    tcp_position2transform(ref_position,&temp);
    transform2RPY_position(&temp,&ref_rpy_pos);

    tcp_position2transform(add_position,&temp);
    transform2RPY_position(&temp,&add_rpy_pos);

    output_rpy_pos.Yaw = ref_rpy_pos.Yaw + add_rpy_pos.Yaw;
    output_rpy_pos.Pitch = ref_rpy_pos.Pitch + add_rpy_pos.Pitch;
    output_rpy_pos.Roll = ref_rpy_pos.Roll + add_rpy_pos.Roll;

    //printf("out X: %lf,Y: %lf,Z: %lf \n",output_rpy_pos.X,output_rpy_pos.Y,output_rpy_pos.Z);

    output_rpy_pos.X = clip(output_rpy_pos.X,clip_vector->X_Min,clip_vector->X_Max);
    output_rpy_pos.Y = clip(output_rpy_pos.Y,clip_vector->Y_Min,clip_vector->Y_Max);
    output_rpy_pos.Z = clip(output_rpy_pos.Z,clip_vector->Z_Min,clip_vector->Z_Max);

    output_rpy_pos.Pitch = clip(output_rpy_pos.Pitch,clip_vector->Pitch_Min,clip_vector->Pitch_Max);
    output_rpy_pos.Roll = clip(output_rpy_pos.Roll,clip_vector->Roll_Min,clip_vector->Roll_Max);
    output_rpy_pos.Yaw = clip(output_rpy_pos.Yaw,clip_vector->Yaw_Min,clip_vector->Yaw_Max);

     if(output_rpy_pos.Yaw<-DK_PI)//만약 회전변화가 180deg 이상이면 => 로봇은 endpoint에 장착된 tool같은것 때문에 제약이 있을 수 있기때문에...
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw+DK_2PI);//위상 반전
    }
    else if (output_rpy_pos.Yaw>DK_PI)
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw-DK_2PI);
    }

    if(output_rpy_pos.Pitch<-DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch+DK_2PI);
    }
    else if (output_rpy_pos.Pitch>DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch-DK_2PI);
    }

    if(output_rpy_pos.Roll<-DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll+DK_2PI);
    }
    else if (output_rpy_pos.Roll>DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll-DK_2PI);
    }

    
    RPY_position2transform(&output_rpy_pos,&temp);
    transform2tcp_position(&temp,output_position);
}

void DK_UR5e_Robot::rpy_position_linear_subtract(UR_RPY_Position* ref_position, UR_RPY_Position* sub_position, UR_RPY_Position* output_position)
{
    UR_RPY_Position ref_rpy_pos;
    UR_RPY_Position sub_rpy_pos;
    UR_RPY_Position output_rpy_pos;
    
    output_rpy_pos.X = (ref_position->X - sub_position->X);
    output_rpy_pos.Y = (ref_position->Y - sub_position->Y);
    output_rpy_pos.Z = (ref_position->Z - sub_position->Z);

    ref_rpy_pos = (*ref_position);
    sub_rpy_pos = (*sub_position);

    output_rpy_pos.Yaw = ref_rpy_pos.Yaw - sub_rpy_pos.Yaw;
    output_rpy_pos.Pitch = ref_rpy_pos.Pitch - sub_rpy_pos.Pitch;
    output_rpy_pos.Roll = ref_rpy_pos.Roll - sub_rpy_pos.Roll;

    // printf("Yaw: %4.5lf, Pitch: %4.5lf, Roll: %4.5lf \n",output_rpy_pos.Yaw,output_rpy_pos.Pitch,output_rpy_pos.Roll);
    if(output_rpy_pos.Yaw<-DK_PI)//만약 회전변화가 180deg 이상이면 => 로봇은 endpoint에 장착된 tool같은것 때문에 제약이 있을 수 있기때문에...
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw+DK_2PI);//위상 반전
    }
    else if (output_rpy_pos.Yaw>DK_PI)
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw-DK_2PI);
    }

    if(output_rpy_pos.Pitch<-DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch+DK_2PI);
    }
    else if (output_rpy_pos.Pitch>DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch-DK_2PI);
    }

    if(output_rpy_pos.Roll<-DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll+DK_2PI);
    }
    else if (output_rpy_pos.Roll>DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll-DK_2PI);
    }

    (*output_position)=output_rpy_pos;
}

void DK_UR5e_Robot::tcp_position_linear_subtract(UR_RPY_Position* ref_position, UR_RPY_Position* sub_position, UR_TCP_Position* output_position)
{
    UR_RPY_Position ref_rpy_pos;
    UR_RPY_Position add_rpy_pos;
    UR_RPY_Position sub_rpy_pos;
    UR_RPY_Position output_rpy_pos;
    transform_M temp;
    output_rpy_pos.X = ref_position->X - sub_position->X;
    output_rpy_pos.Y = ref_position->Y - sub_position->Y;
    output_rpy_pos.Z = ref_position->Z - sub_position->Z;

    ref_rpy_pos = (*ref_position);
    sub_rpy_pos = (*sub_position);

    output_rpy_pos.Yaw = ref_rpy_pos.Yaw - sub_rpy_pos.Yaw;
    output_rpy_pos.Pitch = ref_rpy_pos.Pitch - sub_rpy_pos.Pitch;
    output_rpy_pos.Roll = ref_rpy_pos.Roll - sub_rpy_pos.Roll;

    // printf("Yaw: %4.5lf, Pitch: %4.5lf, Roll: %4.5lf \n",output_rpy_pos.Yaw,output_rpy_pos.Pitch,output_rpy_pos.Roll);
    if(output_rpy_pos.Yaw<-DK_PI)//만약 회전변화가 180deg 이상이면 => 로봇은 endpoint에 장착된 tool같은것 때문에 제약이 있을 수 있기때문에...
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw+DK_2PI);//위상 반전
    }
    else if (output_rpy_pos.Yaw>DK_PI)
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw-DK_2PI);
    }

    if(output_rpy_pos.Pitch<-DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch+DK_2PI);
    }
    else if (output_rpy_pos.Pitch>DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch-DK_2PI);
    }

    if(output_rpy_pos.Roll<-DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll+DK_2PI);
    }
    else if (output_rpy_pos.Roll>DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll-DK_2PI);
    }

    //printf("Yaw: %4.5lf, Pitch: %4.5lf, Roll: %4.5lf \n",output_rpy_pos.Yaw,output_rpy_pos.Pitch,output_rpy_pos.Roll);
    RPY_position2transform(&output_rpy_pos,&temp);
    transform2tcp_position(&temp,output_position);
    //printf("RX: %4.5lf, RY: %4.5lf, RZ: %4.5lf \n",output_position->RX,output_position->RY,output_position->RZ);
}

void DK_UR5e_Robot::tcp_position_linear_subtract(UR_TCP_Position* ref_position, UR_RPY_Position* sub_position, UR_TCP_Position* output_position)
{
    UR_RPY_Position ref_rpy_pos;
    UR_RPY_Position sub_rpy_pos;
    UR_RPY_Position output_rpy_pos;
    transform_M temp;
    output_rpy_pos.X = ref_position->X - sub_position->X;
    output_rpy_pos.Y = ref_position->Y - sub_position->Y;
    output_rpy_pos.Z = ref_position->Z - sub_position->Z;

    tcp_position2transform(ref_position,&temp);
    transform2RPY_position(&temp,&ref_rpy_pos);

    sub_rpy_pos = (*sub_position);

    output_rpy_pos.Yaw = ref_rpy_pos.Yaw - sub_rpy_pos.Yaw;
    output_rpy_pos.Pitch = ref_rpy_pos.Pitch - sub_rpy_pos.Pitch;
    output_rpy_pos.Roll = ref_rpy_pos.Roll - sub_rpy_pos.Roll;

    // printf("Yaw: %4.5lf, Pitch: %4.5lf, Roll: %4.5lf \n",output_rpy_pos.Yaw,output_rpy_pos.Pitch,output_rpy_pos.Roll);
    if(output_rpy_pos.Yaw<-DK_PI)//만약 회전변화가 180deg 이상이면 => 로봇은 endpoint에 장착된 tool같은것 때문에 제약이 있을 수 있기때문에...
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw+DK_2PI);//위상 반전
    }
    else if (output_rpy_pos.Yaw>DK_PI)
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw-DK_2PI);
    }

    if(output_rpy_pos.Pitch<-DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch+DK_2PI);
    }
    else if (output_rpy_pos.Pitch>DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch-DK_2PI);
    }

    if(output_rpy_pos.Roll<-DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll+DK_2PI);
    }
    else if (output_rpy_pos.Roll>DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll-DK_2PI);
    }

    //printf("Yaw: %4.5lf, Pitch: %4.5lf, Roll: %4.5lf \n",output_rpy_pos.Yaw,output_rpy_pos.Pitch,output_rpy_pos.Roll);
    RPY_position2transform(&output_rpy_pos,&temp);
    transform2tcp_position(&temp,output_position);
    //printf("RX: %4.5lf, RY: %4.5lf, RZ: %4.5lf \n",output_position->RX,output_position->RY,output_position->RZ);
}
void DK_UR5e_Robot::tcp_position_linear_subtract(UR_TCP_Position* ref_position, UR_TCP_Position* sub_position, UR_TCP_Position* output_position)
{
    UR_RPY_Position ref_rpy_pos;
    UR_RPY_Position sub_rpy_pos;
    UR_RPY_Position output_rpy_pos;
    transform_M temp;
    output_rpy_pos.X = ref_position->X - sub_position->X;
    output_rpy_pos.Y = ref_position->Y - sub_position->Y;
    output_rpy_pos.Z = ref_position->Z - sub_position->Z;

    tcp_position2transform(ref_position,&temp);
    transform2RPY_position(&temp,&ref_rpy_pos);

    tcp_position2transform(sub_position,&temp);
    transform2RPY_position(&temp,&sub_rpy_pos);

    output_rpy_pos.Yaw = ref_rpy_pos.Yaw - sub_rpy_pos.Yaw;
    output_rpy_pos.Pitch = ref_rpy_pos.Pitch - sub_rpy_pos.Pitch;
    output_rpy_pos.Roll = ref_rpy_pos.Roll - sub_rpy_pos.Roll;

    // printf("Yaw: %4.5lf, Pitch: %4.5lf, Roll: %4.5lf \n",output_rpy_pos.Yaw,output_rpy_pos.Pitch,output_rpy_pos.Roll);
    if(output_rpy_pos.Yaw<-DK_PI)//만약 회전변화가 180deg 이상이면 => 로봇은 endpoint에 장착된 tool같은것 때문에 제약이 있을 수 있기때문에...
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw+DK_2PI);//위상 반전
    }
    else if (output_rpy_pos.Yaw>DK_PI)
    {
        output_rpy_pos.Yaw=(output_rpy_pos.Yaw-DK_2PI);
    }

    if(output_rpy_pos.Pitch<-DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch+DK_2PI);
    }
    else if (output_rpy_pos.Pitch>DK_PI)
    {
        output_rpy_pos.Pitch=(output_rpy_pos.Pitch-DK_2PI);
    }

    if(output_rpy_pos.Roll<-DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll+DK_2PI);
    }
    else if (output_rpy_pos.Roll>DK_PI)
    {
        output_rpy_pos.Roll=(output_rpy_pos.Roll-DK_2PI);
    }

    //printf("Yaw: %4.5lf, Pitch: %4.5lf, Roll: %4.5lf \n",output_rpy_pos.Yaw,output_rpy_pos.Pitch,output_rpy_pos.Roll);
    RPY_position2transform(&output_rpy_pos,&temp);
    transform2tcp_position(&temp,output_position);
    //printf("RX: %4.5lf, RY: %4.5lf, RZ: %4.5lf \n",output_position->RX,output_position->RY,output_position->RZ);

}

void DK_UR5e_Robot::Set_Ethernet_profile_servoj(UR_TCP_Position& from_tcp_pos,UR_TCP_Position& to_tcp_pos,float max_v,float max_w)
{
    // max_vel unit is (m/s);
    //printf("from_tcp_pos.X : %lf, from_tcp_pos.Y: %lf, from_tcp_pos.Z: %lf\n",from_tcp_pos.X,from_tcp_pos.Y,from_tcp_pos.Z);
    //printf("to_tcp_pos.X : %lf, to_tcp_pos.Y: %lf, to_tcp_pos.Z: %lf\n",to_tcp_pos.X,to_tcp_pos.Y,to_tcp_pos.Z);
    //printf("here1\n");
    if(is_Ethernet_Realtime_Server_open == false || max_v<=0.0f || max_w<=0.0f){return;}
    //printf("here2\n");
    double dX = to_tcp_pos.X - from_tcp_pos.X;
    double dY = to_tcp_pos.Y - from_tcp_pos.Y;
    double dZ = to_tcp_pos.Z - from_tcp_pos.Z;
    //printf("dX : %lf, dY: %lf, dZ: %lf\n",dX,dY,dZ);
    UR_RPY_Position to_rpy_pos;
    UR_RPY_Position from_rpy_pos;
    transform_M temp;
    UR_TCP_Position temp_tcp;
    UR_RPY_Position temp_rpy;
    
    tcp_position2transform(&to_tcp_pos,&temp);
    transform2RPY_position(&temp,&to_rpy_pos);

    tcp_position2transform(&from_tcp_pos,&temp);
    transform2RPY_position(&temp,&from_rpy_pos);

    double dRX = to_rpy_pos.Yaw - from_rpy_pos.Yaw;
    double dRY = to_rpy_pos.Pitch - from_rpy_pos.Pitch;
    double dRZ = to_rpy_pos.Roll - from_rpy_pos.Roll;

    if(dRX<-DK_PI)//만약 회전변화가 180deg 이상이면 => 로봇은 endpoint에 장착된 tool같은것 때문에 제약이 있을 수 있기때문에...
    {
        dRX=(dRX+DK_2PI);//위상 반전
    }
    else if (dRX>DK_PI)
    {
        dRX=(dRX-DK_2PI);
    }

    if(dRY<-DK_PI)
    {
        dRY=(dRY+DK_2PI);
    }
    else if (dRY>DK_PI)
    {
        dRY=(dRY-DK_2PI);
    }

    if(dRZ<-DK_PI)
    {
        dRZ=(dRZ+DK_2PI);
    }
    else if (dRZ>DK_PI)
    {
        dRZ=(dRZ-DK_2PI);
    }

    double XYZ_r = sqrt((dX*dX)+(dY*dY)+(dZ*dZ));
    double Rotate_r = sqrt((dRX*dRX)+(dRY*dRY)+(dRZ*dRZ));

    double XYZ_theta = 0.0;
    double XYZ_pi = 0.0;

    double Rotate_theta = 0.0;
    double Rotate_pi = 0.0;

    double frequncy = 2000.0;//Hz => 로봇 제어 hz
    //printf("here3\n");
    //printf("XYZ_r : %lf, Rotate_r: %lf\n",XYZ_r,Rotate_r);
    
    if(XYZ_r <= 0.1)// 움직임의 변동폭이 0.1mm이하면 => 움직임이 정지상태이면
    {   
        XYZ_r=0.0;
        XYZ_theta = 0.0;
        XYZ_pi = 0.0;
    }
    else
    {
        XYZ_theta = acos(dZ/XYZ_r);
        XYZ_pi = atan2(dY,dX);    
    }

    if(Rotate_r <= 0.001)//회전의 변동폭이 0.001rad이하면 => 회전이 정지상태이면
    {
        Rotate_r=0.0;
        Rotate_theta = 0.0;
        Rotate_pi = 0.0;
    }
    else
    {
        Rotate_theta = acos(dRZ/Rotate_r);
        Rotate_pi =  atan2(dRY,dRX);
    }

    int XYZ_step = (int)( (XYZ_r*frequncy) / (max_v*1000.0) );
    double XYZ_step_d = ( (XYZ_r*frequncy) / (max_v*1000.0) );
    int Rotate_step = (int)( (Rotate_r*frequncy) / (max_w) );
    double Rotate_step_d =  ( (Rotate_r*frequncy) / (max_w) );
    
    int step = 0;
    if(XYZ_step>Rotate_step)
    {
        step = XYZ_step;

    }else{
        step = Rotate_step;
    }

    UR_TCP_Position now_tcp_pos;
    now_tcp_pos=from_tcp_pos;
    //printf("here4\n");
    //printf("XYZ_step : %d, Rotate_step: %d, step: %d\n",XYZ_step,Rotate_step,step);

    for(int i_=1;(i_<=step) && is_Ethernet_Realtime_Server_open == true;i_++)
    {   

        if(i_<=XYZ_step)
        {
            //printf("XYZ_theta : %lf, XYZ_pi: %lf\n",XYZ_theta,XYZ_pi);
            now_tcp_pos.X = from_tcp_pos.X + (XYZ_r*( (i_) / (XYZ_step_d) )*sin(XYZ_theta)*cos(XYZ_pi));
            now_tcp_pos.Y = from_tcp_pos.Y + (XYZ_r*( (i_) / (XYZ_step_d) )*sin(XYZ_theta)*sin(XYZ_pi));
            now_tcp_pos.Z = from_tcp_pos.Z + (XYZ_r*( (i_) / (XYZ_step_d) )*cos(XYZ_theta));
        }

        if(i_<=Rotate_step)
        {
            temp_rpy.Yaw= from_rpy_pos.Yaw + (Rotate_r*( (i_) / (Rotate_step_d) )*sin(Rotate_theta)*cos(Rotate_pi));
            temp_rpy.Pitch= from_rpy_pos.Pitch + (Rotate_r*( (i_) / (Rotate_step_d) )*sin(Rotate_theta)*sin(Rotate_pi));
            temp_rpy.Roll= from_rpy_pos.Roll + (Rotate_r*( (i_) / (Rotate_step_d) )*cos(Rotate_theta));
            RPY_position2transform(&temp_rpy,&temp);
            transform2tcp_position(&temp,&temp_tcp);
            now_tcp_pos.RX = temp_tcp.RX;
            now_tcp_pos.RY = temp_tcp.RY;
            now_tcp_pos.RZ = temp_tcp.RZ;
        }

        this->Set_UR_Control(&now_tcp_pos,1);
        
        std::this_thread::sleep_for(std::chrono::microseconds((int)( ( (1.0/frequncy) *1000000.0) / 4.0 ) ));   // 125 us

    }

    this->Set_UR_Control(&to_tcp_pos,1);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    
    //printf("here5\n");
}

void DK_UR5e_Robot::Set_Ethernet_servoj(UR_TCP_Position* tcp_position,float timeout)
{
    if(is_Ethernet_connect == false || is_Ethernet_Realtime_Server_open == true){return;}
    
    std::stringstream stream;
    stream <<"servoj(get_inverse_kin(p[" << std::fixed<<std::setprecision(6) <<tcp_position->X/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->Y/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->Z/1000.0<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RX<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RY<<","
    << std::fixed<< std::setprecision(6) <<tcp_position->RZ<<"]),"
    << "t="<< std::fixed<< std::setprecision(4) <<timeout<<")\n";
    
    std::string command = stream.str();
    char const *pchar = command.c_str();
    send(Ethernet_sock,pchar,command.length(),0);
    std::cout << pchar<<std::endl;
}

void DK_UR5e_Robot::Set_Ethernet_movej(UR_TCP_Position* tcp_position,float max_vel,float max_acc)
{
    if(is_Ethernet_connect == false || is_Ethernet_Realtime_Server_open == true){return;}

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
    char const *pchar = command.c_str();
    

    send(Ethernet_sock,pchar,command.length(),0);
    std::cout << pchar<<std::endl;
}

void DK_UR5e_Robot::Set_Ethernet_movel(UR_TCP_Position* tcp_position,float max_vel,float max_acc)
{
    if(is_Ethernet_connect == false || is_Ethernet_Realtime_Server_open == true){return;}

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
    char const *pchar = command.c_str();
    

    send(Ethernet_sock,pchar,command.length(),0);
    std::cout << pchar<<std::endl;
}


/*void DK_UR5e_Robot::Get_modbus_TCP_Position(UR_TCP_Position* tcp_position)
{
    if(modbus_is_connect==false){return;}
    
    UR_MODBUS_HEX_UNIT unit1(PM_THOUSAND);
    UR_MODBUS_HEX_UNIT unit2(PM_TEN);

    // tcp_x
    if(Get_modbus((char*)&tcp_x,&tcp_position->X,unit2)==false){return;}
    
    // tcp_y
    if(Get_modbus((char*)&tcp_y,&tcp_position->Y,unit2)==false){return;}

    // tcp_z
    if(Get_modbus((char*)&tcp_z,&tcp_position->Z,unit2)==false){return;}

    // tcp_rx
    if(Get_modbus((char*)&tcp_rx,&tcp_position->RX,unit1)==false){return;}

    // tcp_ry
    if(Get_modbus((char*)&tcp_ry,&tcp_position->RY,unit1)==false){return;}

    // tcp_rz
    if(Get_modbus((char*)&tcp_rz,&tcp_position->RZ,unit1)==false){return;}
}*/

void DK_UR5e_Robot::uint16todouble(uint8_t* low,uint8_t* high,double* out_data,double divide=1.0)
{
    uint16_t row = 0;
    ((uint8_t*)&row)[0]=*high;
    ((uint8_t*)&row)[1]=*low;

    if(row > 32767)
    {
        row = 65535-row;
        (*out_data) = (double)(row)/divide*(-1);
        return;
        
    }else
    {
        (*out_data) = (double)(row)/divide;
        return;
    }
}


void DK_UR5e_Robot::Get_modbus_TCP_Position_ALL(UR_TCP_Position* tcp_position=nullptr)
{
    if(modbus_is_connect==false){modbus_is_connect=false;modbus_close();printf("modbus close\n");return;}

    send(this->modbus_socket,(char*)&tcp_position_ALL,12,0);
    modbus_valread = recv(this->modbus_socket,modbus_buffer,21,0);
    if(modbus_valread==-1){return;}
    uint16todouble(modbus_buffer+9,modbus_buffer+10,&(tcp_position->X),10.0);
    uint16todouble(modbus_buffer+11,modbus_buffer+12,&(tcp_position->Y),10.0);
    uint16todouble(modbus_buffer+13,modbus_buffer+14,&(tcp_position->Z),10.0);
    uint16todouble(modbus_buffer+15,modbus_buffer+16,&(tcp_position->RX),1000.0);
    uint16todouble(modbus_buffer+17,modbus_buffer+18,&(tcp_position->RY),1000.0);
    uint16todouble(modbus_buffer+19,modbus_buffer+20,&(tcp_position->RZ),1000.0);
}





/*void DK_UR5e_Robot::Get_modbus_joint(UR_Joint_Position* joint_array_rad)
{
    if(modbus_is_connect==false){return;}
    
    UR_MODBUS_HEX_UNIT unit(PM_THOUSAND);

    // base
    if(Get_modbus((char*)&base,&joint_array_rad->base_rad,unit)==false){return;}
    
    // shoulder
    if(Get_modbus((char*)&shoulder,&joint_array_rad->shoulder_rad,unit)==false){return;}

    // elbow
    if(Get_modbus((char*)&elbow,&joint_array_rad->elbow_rad,unit)==false){return;}

    // wrist1
    if(Get_modbus((char*)&wrist1,&joint_array_rad->wrist1_rad,unit)==false){return;}

    // wrist2
    if(Get_modbus((char*)&wrist2,&joint_array_rad->wrist2_rad,unit)==false){return;}

    // wrist3
    if(Get_modbus((char*)&wrist3,&joint_array_rad->wrist3_rad,unit)==false){return;}
}*/


void DK_UR5e_Robot::Get_modbus_joint_ALL(UR_Joint_Position* joint_array_rad)
{
    if(modbus_is_connect==false){modbus_is_connect=false;modbus_close();printf("modbus close\n");return;}
    send(this->modbus_socket,(char*)&joint_position_ALL,12,0);
    modbus_valread = recv(this->modbus_socket,modbus_buffer,21,0);
    if(modbus_valread==-1){return;}
    uint16todouble(modbus_buffer+9,modbus_buffer+10,&(joint_array_rad->base_rad),1000.0);
    uint16todouble(modbus_buffer+11,modbus_buffer+12,&(joint_array_rad->shoulder_rad),1000.0);
    uint16todouble(modbus_buffer+13,modbus_buffer+14,&(joint_array_rad->elbow_rad),1000.0);
    uint16todouble(modbus_buffer+15,modbus_buffer+16,&(joint_array_rad->wrist1_rad),1000.0);
    uint16todouble(modbus_buffer+17,modbus_buffer+18,&(joint_array_rad->wrist2_rad),1000.0);
    uint16todouble(modbus_buffer+19,modbus_buffer+20,&(joint_array_rad->wrist3_rad),1000.0);    
}

// void DK_UR5e_Robot::Get_modbus_joint_status_ALL(UR_Joint_Position* joint_array_rad,UR_Joint_Current* joint_array_A)
// {
//     if(modbus_is_connect==false){modbus_is_connect=false;modbus_close();printf("modbus close\n");return;}
//     send(this->modbus_socket,(char*)&joint_status_ALL,12,0);
//     modbus_valread = recv(this->modbus_socket,(void*)modbus_buffer,61,0);
//     if(modbus_valread==-1){return;}
//     uint16todouble(modbus_buffer+9,modbus_buffer+10,&(joint_array_rad->base_rad),1000.0);
//     uint16todouble(modbus_buffer+11,modbus_buffer+12,&(joint_array_rad->shoulder_rad),1000.0);
//     uint16todouble(modbus_buffer+13,modbus_buffer+14,&(joint_array_rad->elbow_rad),1000.0);
//     uint16todouble(modbus_buffer+15,modbus_buffer+16,&(joint_array_rad->wrist1_rad),1000.0);
//     uint16todouble(modbus_buffer+17,modbus_buffer+18,&(joint_array_rad->wrist2_rad),1000.0);
//     uint16todouble(modbus_buffer+19,modbus_buffer+20,&(joint_array_rad->wrist3_rad),1000.0);

//     uint16todouble(modbus_buffer+49,modbus_buffer+50,&(joint_array_A->base_A),1000.0);
//     uint16todouble(modbus_buffer+51,modbus_buffer+52,&(joint_array_A->shoulder_A),1000.0);
//     uint16todouble(modbus_buffer+53,modbus_buffer+54,&(joint_array_A->elbow_A),1000.0);
//     uint16todouble(modbus_buffer+55,modbus_buffer+56,&(joint_array_A->wrist1_A),1000.0);
//     uint16todouble(modbus_buffer+57,modbus_buffer+58,&(joint_array_A->wrist2_A),1000.0);
//     uint16todouble(modbus_buffer+59,modbus_buffer+60,&(joint_array_A->wrist3_A),1000.0); 
// }   


/*oid DK_UR5e_Robot::Get_modbus_joint_current(UR_Joint_Current* joint_array_A)
{
    if(modbus_is_connect==false){return;}
    UR_MODBUS_HEX_UNIT unit(PM_THOUSAND);
    // base
    if(Get_modbus((char*)&basecurrent,&joint_array_A->base_A,unit)==false){return;}
    
    // shoulder
    if(Get_modbus((char*)&shouldercurrent,&joint_array_A->shoulder_A,unit)==false){return;}

    // elbow
    if(Get_modbus((char*)&elbowcurrent,&joint_array_A->elbow_A,unit)==false){return;}

    // wrist1
    if(Get_modbus((char*)&wrist1current,&joint_array_A->wrist1_A,unit)==false){return;}

    // wrist2
    if(Get_modbus((char*)&wrist2current,&joint_array_A->wrist2_A,unit)==false){return;}

    // wrist3
    if(Get_modbus((char*)&wrist3current,&joint_array_A->wrist3_A,unit)==false){return;}
}*/


void DK_UR5e_Robot::Get_modbus_joint_current_ALL(UR_Joint_Current* joint_array_A)
{
    if(modbus_is_connect==false){modbus_is_connect=false;modbus_close();printf("modbus close\n");return;}
    send(this->modbus_socket,(char*)&current_ALL,12,0);
    modbus_valread = recv(this->modbus_socket,modbus_buffer,21,0);
    if(modbus_valread==-1){printf("modbus problem\n"); return;}
    uint16todouble(modbus_buffer+9,modbus_buffer+10,&(joint_array_A->base_A),1000.0);
    uint16todouble(modbus_buffer+11,modbus_buffer+12,&(joint_array_A->shoulder_A),1000.0);
    uint16todouble(modbus_buffer+13,modbus_buffer+14,&(joint_array_A->elbow_A),1000.0);
    uint16todouble(modbus_buffer+15,modbus_buffer+16,&(joint_array_A->wrist1_A),1000.0);
    uint16todouble(modbus_buffer+17,modbus_buffer+18,&(joint_array_A->wrist2_A),1000.0);
    uint16todouble(modbus_buffer+19,modbus_buffer+20,&(joint_array_A->wrist3_A),1000.0);   
}


float DK_UR5e_Robot::modbus_hex_to_value_f(uint8_t* buf,UR_MODBUS_HEX_UNIT unit)
{
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

double DK_UR5e_Robot::modbus_hex_to_value_d(uint8_t* buf,UR_MODBUS_HEX_UNIT unit)
{
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



bool DK_UR5e_Robot::Get_modbus(const char* REGISTERS,double* get_value,UR_MODBUS_HEX_UNIT type)
{
    if(modbus_is_connect==false){modbus_is_connect=false;modbus_close();printf("modbus close\n");return false;}

    send(this->modbus_socket ,REGISTERS,12,0);
    modbus_valread = recv(this->modbus_socket,modbus_buffer,1024,0);
    if(modbus_valread==-1){return false;}
    
    *get_value=modbus_hex_to_value_d(modbus_buffer,type);
   
    return true;
}

bool DK_UR5e_Robot::Get_modbus(const char* REGISTERS,float* get_value,UR_MODBUS_HEX_UNIT type)
{
    if(modbus_is_connect==false){modbus_is_connect=false;modbus_close();printf("modbus close\n");return false;}

    send(this->modbus_socket ,REGISTERS,12,0);
    modbus_valread = recv(this->modbus_socket,modbus_buffer,1024,0);
    if(modbus_valread==-1){return false;}
    
    *get_value=modbus_hex_to_value_d(modbus_buffer,type);

    return true;
}





