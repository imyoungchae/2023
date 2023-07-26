#include <iostream>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sstream>
#include <pthread.h>
#include <thread>
#include <chrono> 
#include <mutex>
#include <fstream>
#include <iomanip>
#include <HL/hl.h>
#include <stdio.h>

#include "std_msgs/msg/float64_multi_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <hd.h>
#include <hdCompilerConfig.h>
#include <hdDefines.h>
#include <hdDevice.h>
#include <hdExport.h>
#include <hdScheduler.h>


#include <hduError.h>
#include <hduVector.h>
#include <hduMatrix.h> 

#include "lk_exciter.hpp"


using namespace std;

int calibrationStyle;

lk_exciter     ctrl_c;

struct OmniState		//omni state
{
    hduVector3Dd position;      //3X1 position vector 
    hduVector3Dd velocity;
    hduVector3Dd inp_vel1;
    hduVector3Dd inp_vel2;
    hduVector3Dd inp_vel3;
    hduVector3Dd out_vel1;
    hduVector3Dd out_vel2;
    hduVector3Dd out_vel3;
    hduVector3Dd pos_hist1;     //속도를 알기 위한 position history
    hduVector3Dd pos_hist2;
    hduVector3Dd rot;
    hduVector3Dd joints;
    hduVector3Dd force;

    float thetas[7];
    int buttons[2];
    int buttons_prev[2];
    bool lock;
    hduVector3Dd lock_pos;

    double transform[16];
};

class lk_phantom
{
    public:
    std::string omni_name;

    OmniState *state;

    double init_pos[3]={0};
    double init_rot[3][3]={{1,0,0}, (0,1,0), (0,0,1)};		//rotation의 init값
    double pre_pos[3]={0};
    double curr_pos[3]={0};
    double curr_rot[3]={0};
    double dev_info[6]={0};

    bool init_state=false;
    bool init_flag=false;
    bool slave_init=false;
    int pre_button_1=0;
    int pre_button_2=0;

    double pos_scale=1.0;       //move position scale
    double rot_scale=0.3;       //rotate scale

    bool clutch_strState=false;
    bool clutch_endState=true;
    double clpping_pos[3]       = {0.0};
	double clutch_initPos[3]    = {0.0};
	double clutch_prePos[3]     = {0.0};
	double clutch_curPos[3]     = {0.0};

    //limit_position
	double T_MAX[3]    = {+200.0, +200.0, +100.0};
	double T_MIN[3]    = {-200.0, -200.0, -100.0};

    void init(OmniState *s)
    {
        state=s;
        state->buttons[0] = 0;
		state->buttons[1] = 0;
		state->buttons_prev[0] = 0;
		state->buttons_prev[1] = 0;
		hduVector3Dd zeros(0, 0, 0);
		state->velocity = zeros;
		state->inp_vel1 = zeros;  //3x1 history of velocity
		state->inp_vel2 = zeros;  //3x1 history of velocity
		state->inp_vel3 = zeros;  //3x1 history of velocity
		state->out_vel1 = zeros;  //3x1 history of velocity
		state->out_vel2 = zeros;  //3x1 history of velocity
		state->out_vel3 = zeros;  //3x1 history of velocity
		state->pos_hist1 = zeros; //3x1 history of position
		state->pos_hist2 = zeros; //3x1 history of position
		state->lock = false;
		state->lock_pos = zeros;
    }

	void init_position()
	{
		printf("\n\n");
		printf("omni initialization\n");

		int cnt=0;
		for (int i=0; i<3; i++)
		{
			clpping_pos[i]=0;
			clutch_curPos[i]=0;
		}

		while (1)
		{
			auto loop_hz=std::chrono::steady_clock::now()+std::chrono::microseconds(500);

			read_initFlag();

			if (init_flag&&slave_init)
        	{
            	cnt++;
            	printf("init_flag and slave_init are both 1, cnt increased: %d\n", cnt); // 이 코드를 추가합니다.
        	}

			if (cnt>=1000)
			{
				//get init position
				init_pos[0]=state->transform[12];
				init_pos[1]=state->transform[13];
				init_pos[2]=state->transform[14];

				//get init rotation
				double temp_ori[3][3]={{0}};
				temp_ori[0][0]=state->transform[0];
				temp_ori[0][1]=state->transform[1];
				temp_ori[0][2]=state->transform[2];
				temp_ori[1][0]=state->transform[4];
				temp_ori[1][1]=state->transform[5];
				temp_ori[1][2]=state->transform[6];
				temp_ori[2][0]=state->transform[8];
				temp_ori[2][1]=state->transform[9];
				temp_ori[2][2]=state->transform[10];

				//rotation vector 로 orientation vector 를 얻기 위해 inverse(transpose)
				for(int i=0; i<3; i++)
				{
					for(int j=0; j<3; j++)
					{
						init_rot[j][i]=temp_ori[i][j];
						printf("init_rot[%d][%d]=temp_ori[%d][%d]\n",j,i,i,j);
					}
				}
				
				printf("init_rot: %.2f, %.2f, %.2f\n", init_rot[0][0], init_rot[0][1], init_rot[0][2]);
				printf("init_rot: %.2f, %.2f, %.2f\n", init_rot[1][0], init_rot[1][1], init_rot[1][2]);
				printf("init_rot: %.2f, %.2f, %.2f\n", init_rot[2][0], init_rot[2][1], init_rot[2][2]);
				printf("=============================\n");
				printf("temp_ori: %.2f, %.2f, %.2f\n", temp_ori[0][0], temp_ori[0][1], temp_ori[0][2]);
				printf("temp_ori: %.2f, %.2f, %.2f\n", temp_ori[1][0], temp_ori[1][1], temp_ori[1][2]);
				printf("temp_ori: %.2f, %.2f, %.2f\n", temp_ori[2][0], temp_ori[2][1], temp_ori[2][2]);

				break;
			}

			std::this_thread::sleep_until(loop_hz);
		}

		printf("init clear!\n");
	}
	void read_initFlag()
	{
		// 동시에 버튼 1과 버튼 2가 눌렀을 때 init_flag와 slave_init을 1로 변경
		if (state->buttons[1] == 1 && state->buttons[2] == 1)
		{
			init_flag = 1;
			slave_init = 1;

		}
		//printf("init_flag: %d, slave_init: %d\n", init_flag, slave_init);
		clutch_strState = state->buttons[0];
	}
	
    void get_position()
    {
        pre_pos[0]=state->transform[12];        //transform matrix 의 position vector 자리에다가 집어넣기
        pre_pos[1]=state->transform[13];
        pre_pos[2]=state->transform[14];

        //position zero setting
        for (int i=0; i<3; i++)
        {
            pre_pos[i]-=init_pos[i];    //0점 만들어주기
            pre_pos[i]*=pos_scale;

            curr_pos[i]=pre_pos[i]-clpping_pos[i]-clutch_curPos[i];
        }

        //limit position
		for (int i = 0; i < 3; i++)
        {
			if (curr_pos[i] > T_MAX[i])
            {
				clpping_pos[i]	= pre_pos[i] - clutch_curPos[i] - T_MAX[i];
				curr_pos[i]		= T_MAX[i];
			}
			else if (curr_pos[i] < T_MIN[i])
            {
				clpping_pos[i]	= pre_pos[i] - clutch_curPos[i] - T_MIN[i];
				curr_pos[i]		= T_MIN[i];
			}
		}       
    }

    void get_rotation()
    {
        //등가각축 공식
        double temp_theta=0;
        double temp_K=0;
        double temp_ori[3][3]={{0}};
        double dev_ori[3][3]={{0}};

        //get rotation matrix
        temp_ori[0][0]	= state->transform[0];
		temp_ori[0][1]	= state->transform[1];
		temp_ori[0][2]	= state->transform[2];
		temp_ori[1][0]	= state->transform[4];
		temp_ori[1][1]	= state->transform[5];
		temp_ori[1][2]	= state->transform[6];
		temp_ori[2][0]	= state->transform[8];
		temp_ori[2][1]	= state->transform[9];
		temp_ori[2][2]	= state->transform[10];

        // multiple frame init & current
		for (int i = 0; i < 3; i++)
        {
			for (int j = 0; j < 3; j++) 
            {
				dev_ori[i][j] = 0;
				for (int k = 0; k < 3; k++)
				{
					dev_ori[i][j] += init_rot[i][k] * temp_ori[k][j];
					//printf("dev_ori[%d][%d](%.2f)=init_rot[%d][%d](%.2f)*temp_ori[%d][%d](%.2f)\n",i,j,dev_ori[i][j],i,k,init_rot[i][k],k,j,temp_ori[k][j]);
				}
			}
		}

		temp_theta  = acos( (dev_ori[0][0] + dev_ori[1][1] + dev_ori[2][2] - 1) / 2.0 );        //이건 그냥 공식
		temp_K      = 1 / (2.0 * sin(temp_theta)); 
		// prevent overshoot caused by 'theta = 0 -> temp_K = non'
		if(temp_theta == 0) {temp_K = 1;}

		curr_rot[0]	= (dev_ori[2][1] - dev_ori[1][2]) * temp_K * temp_theta;        //방향 K
		curr_rot[1] = (dev_ori[0][2] - dev_ori[2][0]) * temp_K * temp_theta;        //회전량 theta
		curr_rot[2] = (dev_ori[1][0] - dev_ori[0][1]) * temp_K * temp_theta;		

		// update current rotation
		dev_info[3]	= curr_rot[0] * rot_scale;
		dev_info[4]	= curr_rot[1] * rot_scale;
		dev_info[5]	= curr_rot[2] * rot_scale;
    }

    void clutch_process() {

		// end of position holding
		if (!clutch_strState && !clutch_endState)
        {
			std::cout << "end" << std::endl;
			// integrate displacement
			for (int i = 0; i < 3; i++)
            {
				clutch_curPos[i] += clutch_prePos[i];
			}
			// end clutch process
			clutch_endState=true;
		}
		// normal state
		else if (!clutch_strState)
        {
			// input the current position
			dev_info[0]	= curr_pos[0];
			dev_info[1]	= curr_pos[1];
			dev_info[2]	= curr_pos[2];
			// reset the clutching started position
			for (int i = 0; i < 3; i++)
            {
				clutch_initPos[i] = curr_pos[i];
			}
		}

		// start of position holding
		if (clutch_strState)
        {
			std::cout << "hold" << std::endl;
			clutch_endState = false;
				
			for (int i = 0; i < 3; i++) {
				clutch_prePos[i] = (curr_pos[i] - clutch_initPos[i]);
			}
		}
	}
};

HDCallbackCode HDCALLBACK omni_state_callback(void *pUserData)	
{
	hduMatrix omni_matrix;
	double rot_matrix[3][3];
    HHD hhd = hdGetCurrentDevice ();
    double position[3]; 
    double force[] = {0.0, 0.0, 0.0};
    double k = 0.5; // 스프링 상수

	OmniState *omni_state = static_cast<OmniState *>(pUserData);
	if (hdCheckCalibration() == HD_CALIBRATION_NEEDS_UPDATE)
    {
		std::cout << "[DEBUG] Updating calibration.." << '\n';
		hdUpdateCalibration(calibrationStyle);
	}
	hdBeginFrame(hdGetCurrentDevice());

	// get currnet angle, position, force
	hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES, omni_state->rot);
	hdGetDoublev(HD_CURRENT_POSITION, omni_state->position);
	hdGetDoublev(HD_CURRENT_JOINT_ANGLES, omni_state->joints);
	hdGetDoublev(HD_CURRENT_FORCE, omni_state->force);
	hdGetDoublev(HD_CURRENT_TRANSFORM, omni_state->transform);
		hdGetDoublev (HD_CURRENT_POSITION, position);
	    if (position[0] < 0) // 힘 방향 바꾸기
 			force[0] = -k * position[0];  
        hdSetDoublev (HD_CURRENT_FORCE, force);	// Cartesian 좌표 계산0
	// get button state
	int nButtons	= 0;
	hdGetIntegerv(HD_CURRENT_BUTTONS, &nButtons);
	omni_state->buttons[0]	= (nButtons & HD_DEVICE_BUTTON_1) ? 1 : 0;
	omni_state->buttons[1]	= (nButtons & HD_DEVICE_BUTTON_2) ? 1 : 0;
	omni_state->buttons[2] = ((nButtons & (HD_DEVICE_BUTTON_1 | HD_DEVICE_BUTTON_2)) == (HD_DEVICE_BUTTON_1 | HD_DEVICE_BUTTON_2)) ? 1 : 0;

	hdEndFrame(hdGetCurrentDevice());

	HDErrorInfo error;

	if (HD_DEVICE_ERROR(error = hdGetError()))
    {
		hduPrintError(stderr, &error, "Error during main scheduler callback");
		if (hduIsSchedulerError(&error))
			return HD_CALLBACK_DONE;
	}

	float t[7] = { 0., omni_state->joints[0], omni_state->joints[1],
			omni_state->joints[2] - omni_state->joints[1], omni_state->rot[0],
			omni_state->rot[1], omni_state->rot[2] };
	for (int i = 0; i < 7; i++)
		omni_state->thetas[i] = t[i];

	return HD_CALLBACK_CONTINUE;
}

void HHD_Auto_Calibration() {		//calibration

	int supportedCalibrationStyles;
	HDErrorInfo error;

	hdGetIntegerv(HD_CALIBRATION_STYLE, &supportedCalibrationStyles);
	if (supportedCalibrationStyles & HD_CALIBRATION_ENCODER_RESET) {
		calibrationStyle = HD_CALIBRATION_ENCODER_RESET;
		std::cout << "HD_CALIBRATION_ENCODER_RESE.." << '\n';
	}
	if (supportedCalibrationStyles & HD_CALIBRATION_INKWELL) {
		calibrationStyle = HD_CALIBRATION_INKWELL;
		std::cout << "HD_CALIBRATION_INKWELL.." << '\n';
	}
	if (supportedCalibrationStyles & HD_CALIBRATION_AUTO) {
		calibrationStyle = HD_CALIBRATION_AUTO;
		std::cout << "HD_CALIBRATION_AUTO.." << '\n';
	}
	if (calibrationStyle == HD_CALIBRATION_ENCODER_RESET) {
	  do {
		hdUpdateCalibration(calibrationStyle);
		std::cout << "Calibrating.. (put stylus in well)" << '\n';
		if (HD_DEVICE_ERROR(error = hdGetError())) {
			hduPrintError(stderr, &error, "Reset encoders reset failed.");
			break;
		}  
	} while (hdCheckCalibration() != HD_CALIBRATION_OK);
	std::cout << "Calibration complete." << '\n';
	}
	if (hdCheckCalibration() == HD_CALIBRATION_NEEDS_MANUAL_INPUT) {
	  std::cout << "Please place the device into the inkwell for calibration." << '\n';
	}
}

void publish_omni(OmniState *s)
{
    OmniState *state;
    state=s;

    while (ctrl_c.ok())
    {
        auto loop_hz=std::chrono::steady_clock::now()+std::chrono::milliseconds(1);
        std::this_thread::sleep_until(loop_hz);
    }
}

HDCallbackCode HDCALLBACK RenderWallOperation (void* pUserData)
{
    HHD hhd = hdGetCurrentDevice ();
    double position[3]; 
    double force[] = {0.0, 0.0, 0.0};
    double k = 0.5; // 스프링 상수

    hdBeginFrame (hhd); 
        hdGetDoublev (HD_CURRENT_POSITION, position);
        if (position[0] > 0) // 힘 방향 바꾸기
 			force[0] = -k * position[0];  
        hdSetDoublev (HD_CURRENT_FORCE, force);	// Cartesian 좌표 계산0
    hdEndFrame (hhd);
    return HD_CALLBACK_CONTINUE;
}

HDCallbackCode HDCALLBACK GravityWellSelection (void* pUserData)
{
    HHD hhd = hdGetCurrentDevice ();
	hduVector3Dd position;
	hdGetDoublev (HD_CURRENT_POSITION, position);

	hduVector3Dd gravityWellCenter (0, 0, 0);
	const float k = 2.0;
	hduVector3Dd forceVector = (gravityWellCenter-position)*k;
    return HD_CALLBACK_CONTINUE;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("publisher_node");
    auto omni_pub = node->create_publisher<std_msgs::msg::Float64MultiArray>("omni_data", 10);

    HDErrorInfo error;
    HHD hHD;
    hHD=hdInitDevice(HD_DEFAULT_DEVICE); // 디바이스 초기화
    if(HD_DEVICE_ERROR(error=hdGetError()))
    {
        std::cout<<"Failed to initialize haptic device"<<'\n';
        return -1;
    }
    if (HD_DEVICE_ERROR(error = hdGetError())) {
        std::cout << "Failed to start the scheduler" << '\n';
        return -1;
    }
    HHD_Auto_Calibration();
    OmniState	state;
    lk_phantom	omni;
    omni.init(&state);
    hdEnable(HD_FORCE_OUTPUT); // 디바이스 힘 출력 활성화
    hdStartScheduler();
    //hdScheduleAsynchronous(GravityWellSelection,NULL, HD_MAX_SCHEDULER_PRIORITY);
    hdScheduleAsynchronous(omni_state_callback, &state, HD_MAX_SCHEDULER_PRIORITY);
    //hdScheduleAsynchronous(RenderWallOperation, NULL, HD_MAX_SCHEDULER_PRIORITY);
    omni.init_position();

    while (1) 
    {
        omni.get_position();
        omni.get_rotation(); 
        printf("omni now x:%lf\t y:%lf\t z:%lf\t rx:%lf\t ry:%lf\t rz:%lf\n", omni.dev_info[0], omni.dev_info[1], omni.dev_info[2], omni.dev_info[3], omni.dev_info[4], omni.dev_info[5]);
        omni.read_initFlag(); 
        omni.clutch_process();
    
        std_msgs::msg::Float64MultiArray omni_data;
        omni_data.data = {omni.dev_info[0], omni.dev_info[1], omni.dev_info[2], omni.dev_info[3], omni.dev_info[4], omni.dev_info[5]};
        omni_pub->publish(omni_data);
        rclcpp::spin_some(node);
        rclcpp::Rate loop_rate(100); // 100 Hz를 사용하거나 필요에 맞게 조정하세요.
        loop_rate.sleep();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    omni.read_initFlag();
    hdStopScheduler(); // 스케줄러 종료
    hdDisableDevice(hHD); // 디바이스 종료
    
    return 0;
}
