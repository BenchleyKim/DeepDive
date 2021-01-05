# 윈도우 기준 Azure Kinect DK Body Tracking 설치법 

참고링크 : https://docs.microsoft.com/en-us/azure/kinect-dk/set-up-azure-kinect-dk 

## 1. Kinect SDK 설치
* https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md 에서 1.4.0의 MSI와 펌웨어 파일을 받는다.
* MSI 바로 실행해서 설치 
* 키넥트는 초기 펌웨어를 가지고 있기 떄문에 펌웨어 업데이트를 진행한다. 
* SDK설치 위치(보통 C:\\User\Program Files\Azure Kinect SDK v1.4.1)\tools\ 로 접근하면 AzureKinectFrimwareTool.exe와 firmware 폴더가 있는데가 있는데 
* 다운받은 AzureKinectDK_Fw_1.5.926614.bin. 를 firmware 폴더에 넣는다
* 키넥트를 연결하고 명령 프롬프트에 다음 명령어 입력   
`
 AzureKinectFirmwareTool.exe -u 펌웨어 파일 이름 
`   

## 2. 설치 확인   
* SDK설치 위치(보통 C:\Program Files\Azure Kinect SDK v1.4.1)\tools\ 에 있는 `k4aviewer.exe` 실행 
* Open Device 선택하고 > Start
* 실행이 잘되는지 확인한다. 
* 다음으로 Body Tracking SDK를 설치한다. 

## 3. 최신 그래픽 드라이버 설치
* 설치를 시작하기 전에 그래픽 드라이버를 업데이트 해준다. 
* https://www.nvidia.com/Download/index.aspx?lang=en-us 

## 4. Visual Studio 2015 C++설치
* Visual Studio 2015 재배포 패키지 C++ 설치를 해준다.(따로 설정할 필요없이 VS2015 설치하면 자동으로 설치됨)
* 단 Visual Studio 2015가 아닌 2017, 2019가 설치되어 있는 경우 그냥 진행해도 상관없음.

## 5. Body Tracking SDK 설치
* https://docs.microsoft.com/en-us/azure/kinect-dk/body-sdk-download 에서 원하는 버전 msi 다운로드
* msi 파일 실행해서 설치 완료 
## 6. 설치 확인 
* 설치 위치 (보통 C:\Program Files\Azure Kinect Body Tracking SDK)\tools\ 에 접근
* cmd 경로에서 `k4abt_simple_3d_viewer.exe` 입력해서 실행   ( 단 `k4aviewer.exe` 이 켜져 있으면 실행 안됨) 
* 만약 안되면 GPU 문제이기 때문에 cmd에  `k4abt_simple_3d_viewer.exe CPU` 입력해서 실행 확인 

## 7. 개발을 위한 VS에 키넥트 라이브러리 추가
* 빈 프로젝트 생성 `새 프로젝트 만들기 > c++ 콘솔 앱 ` 추천 
* NuGet package 설치 `[참조링크] : https://docs.microsoft.com/en-us/nuget/quickstart/install-and-use-a-package-in-visual-studio `  
* 패키지 매니저 UI에서 우클릭해서 References 에 Manage NuGet Packages 를 클릭한다.
* Browse 탭에서 nuget.org 를 선택해 Microsoft.Azure.Kinect를 검색해서 Sensor와 Body Tracking 라이브러리를 추가해준다.
* Source 폴더에 새 c 파일을 만들어주고 개발을 시작한다. 

## 8. 라이브러리 추가 확인 
* 소스파일에 `main.c` 파일 생성해 주고 다음 코드 입력 (해당 코드는 제품 번호를 출력하고 종료되는 코드이다. )
* 빌드 또는 실행 

```
#pragma comment(lib, "k4a.lib")
#include <k4a/k4a.h>

#include <stdio.h>
#include <stdlib.h>

int main()
{
    uint32_t count = k4a_device_get_installed_count();
    if (count == 0)
    {
        printf("No k4a devices attached!\n");
        return 1;
    }

    // Open the first plugged in Kinect device
    k4a_device_t device = NULL;
    if (K4A_FAILED(k4a_device_open(K4A_DEVICE_DEFAULT, &device)))
    {
        printf("Failed to open k4a device!\n");
        return 1;
    }

    // Get the size of the serial number
    size_t serial_size = 0;
    k4a_device_get_serialnum(device, NULL, &serial_size);

    // Allocate memory for the serial, then acquire it
    char* serial = (char*)(malloc(serial_size));
    k4a_device_get_serialnum(device, serial, &serial_size);
    printf("Opened device: %s\n", serial);
    free(serial);

    // Configure a stream of 4096x3072 BRGA color data at 15 frames per second
    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config.camera_fps = K4A_FRAMES_PER_SECOND_15;
    config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    config.color_resolution = K4A_COLOR_RESOLUTION_3072P;

    // Start the camera with the given configuration
    if (K4A_FAILED(k4a_device_start_cameras(device, &config)))
    {
        printf("Failed to start cameras!\n");
        k4a_device_close(device);
        return 1;
    }

    // Camera capture and application specific code would go here

    // Shut down the camera when finished with application logic
    k4a_device_stop_cameras(device);
    k4a_device_close(device);

    return 0;
}
```



## 7. 100프레임 간(시간 변경 가능) 신체추적후 기록 저장하는 코드 
* `while (frame_count < 100);` 를 수정해서 원하는 시간 설정가능 
* 현재 3000번대 그래픽카드 이슈로 실행 안됨 > 다른 환경에서 되는지 확인해봐야함. 
```
#include <stdio.h>
#include <stdlib.h>

#include <k4a/k4a.h>
#include <k4abt.h>

#define VERIFY(result, error)                                                                            \
    if(result != K4A_RESULT_SUCCEEDED)                                                                   \
    {                                                                                                    \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        exit(1);                                                                                         \
    }                                                                                                    \

int main()
{
    k4a_device_t device = NULL;
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");

    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");

    k4a_calibration_t sensor_calibration;
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensor_calibration),
        "Get depth camera calibration failed!");

    k4abt_tracker_t tracker = NULL;
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker), "Body tracker initialization failed!");

    int frame_count = 0;
    do
    {
        k4a_capture_t sensor_capture;
        k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &sensor_capture, K4A_WAIT_INFINITE);
        if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED)
        {
            frame_count++;
            k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, sensor_capture, K4A_WAIT_INFINITE);
            k4a_capture_release(sensor_capture); // Remember to release the sensor capture once you finish using it
            if (queue_capture_result == K4A_WAIT_RESULT_TIMEOUT)
            {
                // It should never hit timeout when K4A_WAIT_INFINITE is set.
                printf("Error! Add capture to tracker process queue timeout!\n");
                break;
            }
            else if (queue_capture_result == K4A_WAIT_RESULT_FAILED)
            {
                printf("Error! Add capture to tracker process queue failed!\n");
                break;
            }

            k4abt_frame_t body_frame = NULL;
            k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);
            if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
            {
                // Successfully popped the body tracking result. Start your processing

                size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
                printf("%zu bodies are detected!\n", num_bodies);

                k4abt_frame_release(body_frame); // Remember to release the body frame once you finish using it
            }
            else if (pop_frame_result == K4A_WAIT_RESULT_TIMEOUT)
            {
                //  It should never hit timeout when K4A_WAIT_INFINITE is set.
                printf("Error! Pop body frame result timeout!\n");
                break;
            }
            else
            {
                printf("Pop body frame result failed!\n");
                break;
            }
        }
        else if (get_capture_result == K4A_WAIT_RESULT_TIMEOUT)
        {
            // It should never hit time out when K4A_WAIT_INFINITE is set.
            printf("Error! Get depth frame time out!\n");
            break;
        }
        else
        {
            printf("Get depth capture returned error: %d\n", get_capture_result);
            break;
        }

    } while (frame_count < 100);

    printf("Finished body tracking processing!\n");

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);
    k4a_device_stop_cameras(device);
    k4a_device_close(device);

    return 0;
}
```




