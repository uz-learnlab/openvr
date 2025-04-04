//========= Copyright Valve Corporation ============//
// compile with: g++ fosfenos_stereo.cpp shared/pathtools.cpp shared/Matrices.cpp -o fosfenos_stereo `sdl2-config --cflags --libs` `pkg-config --cflags --libs opencv4` -I/usr/include/openvr -L/usr/lib/x86_64-linux-gnu -lopenvr_api -I/home/ivea/fosfenos/openvr/samples/shared -lGL -lGLEW -ldl -D_LINUX


#include <SDL.h>
#include <GL/glew.h>
#include <SDL_opengl.h>
#if defined( OSX )
#include <Foundation/Foundation.h>
#include <AppKit/AppKit.h>
#include <OpenGL/glu.h>
// Apple's version of glut.h #undef's APIENTRY, redefine it
#define APIENTRY
#else
#include <GL/glu.h>
#endif
#include <stdio.h>
#include <string>
#include <cstdlib>

#include <openvr.h>

#include "shared/lodepng.h"
#include "shared/Matrices.h"
#include "shared/pathtools.h"

#if defined(POSIX)
#include "unistd.h"
#endif

#ifndef LINUX
#define APIENTRY
#endif

#ifndef _countof
#define _countof(x) (sizeof(x)/sizeof((x)[0]))
#endif

#include <opencv2/opencv.hpp>

#include <fstream>
#include <thread>
#include <chrono>
#include <cassert>
#include <iostream>
using namespace cv;
using namespace std;
using namespace std::chrono;

void ThreadSleep( unsigned long nMilliseconds )
{
#if defined(_WIN32)
	::Sleep( nMilliseconds );
#elif defined(POSIX)
	usleep( nMilliseconds * 1000 );
#endif
}

class CGLRenderModel
{
public:
	CGLRenderModel( const std::string & sRenderModelName );
	~CGLRenderModel();

	bool BInit( const vr::RenderModel_t & vrModel, const vr::RenderModel_TextureMap_t & vrDiffuseTexture );
	void Cleanup();
	const std::string & GetName() const { return m_sModelName; }

private:
	GLuint m_glVertBuffer;
	GLuint m_glIndexBuffer;
	GLuint m_glVertArray;
	GLuint m_glTexture;
	GLsizei m_unVertexCount;
	std::string m_sModelName;
};

static bool g_bPrintf = true;

//-----------------------------------------------------------------------------
// Purpose:
//------------------------------------------------------------------------------
class CMainApplication
{
public:
	CMainApplication( int argc, char *argv[] );
	virtual ~CMainApplication();

	bool BInit();

	void Shutdown();

	void RunMainLoop();
	bool HandleInput();
	void RenderFrame();

	void UpdateHMDMatrixPose();

	Matrix4 ConvertSteamVRMatrixToMatrix4( const vr::HmdMatrix34_t &matPose );

	// Phosphenes functions
	bool Initialitation();
	bool ImageIntoPhosohenes();
	
private: 
	bool m_bDebugOpenGL;
	bool m_bVerbose;
	bool m_bPerf;
	bool m_bVblank;
	bool m_bGlFinishHack;

	bool iniciogl;
	GLuint image_tex_Left;
	GLuint image_tex_Right;

	vr::IVRSystem *m_pHMD;
	std::string m_strDriver;
	std::string m_strDisplay;
	vr::TrackedDevicePose_t m_rTrackedDevicePose[ vr::k_unMaxTrackedDeviceCount ];
	Matrix4 m_rmat4DevicePose[ vr::k_unMaxTrackedDeviceCount ];
	

private: // SDL bookkeeping
	SDL_Window *m_pCompanionWindow;
	uint32_t m_nCompanionWindowWidth;
	uint32_t m_nCompanionWindowHeight;

	SDL_GLContext m_pContext;

	private: // Phosphenes

	uint32_t				m_nLastFrameSequence = 0;
	cv::Mat                 image, img_ph;

	Mat plotImg, sprite, reference_sprite;
	Mat K;
	int k, w, f, deltaX, deltaY;
	double fx, fy;
	int N_levels;    //From 0 to 7
	int x0, y0;
	int X0, Y0;
	int grid_mode = 1;
	int N_fos = 1024;
	double noise_stddev = 0.0;
	double rExt;
	cv::Mat phosphenesPos;  // The actual pixel position of each phosphene
	cv::Mat phosphenesPos_vis; // The actual pixel position of each phosphene for visualization
	cv::Mat XiCam;          // Rays emanating from each phosphene
	std::vector<cv::Mat> sprites;
	double size_sprite;

	cv::Mat srcbn1Right, srcbn1Left;
	cv::Mat ampVectorRight;

	std::vector<cv::Rect> roi;
	std::vector<std::string> window_name;

	int TemporalMode = 0;
	float x_right;

	//Temporal model
	cv::Mat LUT, mat, casoRight, casoLeft, matriz_imagen, imgRight; 
	unsigned int grayThreshold, grayThreshold1, grayThreshold2; 
	double minVal, maxVal;

private: // OpenGL bookkeeping
	int m_iValidPoseCount;
	bool m_bShowCubes;

	std::string m_strPoseClasses;                            // what classes we saw poses for this frame
	char m_rDevClassChar[ vr::k_unMaxTrackedDeviceCount ];   // for each device, a character representing its class
	
	int m_iSceneVolumeInit;                                  // if you want something other than the default 20x20x20
	
	GLuint m_glSceneVertBuffer;
	GLuint m_unSceneVAO;
	GLuint m_unCompanionWindowVAO;

	GLuint m_unControllerVAO;

	Matrix4 m_mat4HMDPose;

	GLuint m_unSceneProgramID;
	GLuint m_unCompanionWindowProgramID;
	GLuint m_unControllerTransformProgramID;
	GLuint m_unRenderModelProgramID;


	struct FramebufferDesc
	{
		GLuint m_nDepthBufferId;
		GLuint m_nRenderTextureId;
		GLuint m_nRenderFramebufferId;
		GLuint m_nResolveTextureId;
		GLuint m_nResolveFramebufferId;
	};
	FramebufferDesc leftEyeDesc;
	FramebufferDesc rightEyeDesc;

	std::vector< CGLRenderModel * > m_vecRenderModels;

	vr::VRActionHandle_t m_actionHideCubes = vr::k_ulInvalidActionHandle;
	vr::VRActionHandle_t m_actionHideThisController = vr::k_ulInvalidActionHandle;
	vr::VRActionHandle_t m_actionTriggerHaptic = vr::k_ulInvalidActionHandle;
	vr::VRActionHandle_t m_actionAnalongInput = vr::k_ulInvalidActionHandle;

	vr::VRActionSetHandle_t m_actionsetDemo = vr::k_ulInvalidActionSetHandle;

	cv::VideoCapture m_cam;
};

// -------------------------------------FUNCTIONS FROM PHOSPHENES--------------------
cv::Mat SingleViewToPhosphenes(cv::Mat image, int tpx, int tpy, int sizex = 12 * 32, int sizey = 12 * 32) {
	cv::Rect myROI_R(tpx, tpy, sizex, sizey);
	image = image(myROI_R);
	
	// Converting into BW 
	cv::Mat  capa_img;
	cvtColor(image, capa_img, cv::COLOR_BGR2GRAY);
	return capa_img;
}

int computeDeltaFromNumberOfPhosphenes(int rows, int cols, int N_fos)
{
	int computed_delta = (int)floor(sqrt(double(cols * rows) / double(N_fos)));
	if (computed_delta % 2 == 0)
	{
		return computed_delta - 1;
	}
	else
		return computed_delta;
}

static void genPhosphenesPos(int rows, int cols, double rExt, int x00, int y00, int deltaX, int deltaY, int grid_mode, double noise_stddev, Mat& phosphenesPos, Mat& phosphenesPos_vis, Mat& XiCam, Mat K)
{

	double yIni = std::min(double(y00), rExt);
	double xIni = min(double(x00), rExt);
	int nPointsX = cols / deltaX;
	int nPointsY = rows / deltaY;

	cv::Mat phosPoints_ray = cv::Mat::ones(0, 3, CV_64FC1);
	cv::Mat phosPoints_vis = cv::Mat::ones(0, 3, CV_64FC1);


	double rExt2 = pow(rExt, 2);

	//std::default_random_engine generator;
	//std::normal_distribution<double> distribution(0.0, noise_stddev);

	if (grid_mode == 1)
	{
		for (int kY = 0; kY < nPointsY; kY++)
		{
			for (int kX = 0; kX < nPointsX; kX++)
			{
				double noise_x = 0; // distribution(generator);
				double x = -xIni + deltaX * kX;
				double x_vis = x + noise_x;

				if ((x + x00 - (deltaX - 1) / 2 < 0) || (x + x00 + (deltaX - 1) / 2 >= cols) || (x_vis + x00 - (deltaX - 1) / 2 < 0) || (x_vis + x00 + (deltaX - 1) / 2 >= cols))
					continue;

				double noise_y = 0;// distribution(generator);
				double y = -yIni + deltaY * kY;
				double y_vis = y + noise_y;

				if ((y + y00 - (deltaY - 1) / 2 < 0) || (y + y00 + (deltaY - 1) / 2 >= rows) || (y_vis + y00 - (deltaY - 1) / 2 < 0) || (y_vis + y00 + (deltaY - 1) / 2 >= rows))
					continue;


				double d2 = x * x + y * y;
				double d2_vis = x_vis * x_vis + y_vis * y_vis;
				if (d2 < rExt2 && d2_vis < rExt2)
				{

					cv::Mat vecPoints(cv::Point3d(x + x00, y + y00, 1.0));
					phosPoints_ray.push_back(vecPoints.t());
					cv::Mat vecPoints_vis(cv::Point3d(x_vis + x00, y_vis + y00, 1.0));
					phosPoints_vis.push_back(vecPoints_vis.t());
				}

			}
		}
	}
	else
	{
		bool displaced = false;
		for (int kY = 0; kY < nPointsY; kY++)
		{

			double extra_displacement;
			if (!displaced)
			{
				extra_displacement = 0;
				displaced = true;
			}
			else
			{
				extra_displacement = (deltaX - 1) / 2;
				displaced = false;
			}

			for (int kX = 0; kX < nPointsX; kX++)
			{

				if (!displaced && (kX == nPointsX - 1))
					break;

				double noise_x = 0; // distribution(generator);
				double x = -xIni + deltaX * kX + extra_displacement;
				double x_vis = x + noise_x;

				if ((x + x00 - (deltaX - 1) / 2 < 0) || (x + x00 + (deltaX - 1) / 2 >= 1024) || (x_vis + x00 - (deltaX - 1) / 2 < 0) || (x_vis + x00 + (deltaX - 1) / 2 >= cols))
					continue;

				double noise_y = 0; // distribution(generator);
				double y = -yIni + deltaY * kY;
				double y_vis = y + noise_y;

				if ((y + y00 - (deltaY - 1) / 2 < 0) || (y + y00 + (deltaY - 1) / 2 >= 1024) || (y_vis + y00 - (deltaY - 1) / 2 < 0) || (y_vis + y00 + (deltaY - 1) / 2 >= rows))
					continue;


				double d2 = x * x + y * y;
				double d2_vis = x_vis * x_vis + y_vis * y_vis;

				if (d2 < rExt2 && d2_vis < rExt2)
				{
					cv::Mat vecPoints(cv::Point3d(x + x00, y + y00, 1.0));

					Mat vecPointsT = vecPoints.t();

					phosPoints_ray.push_back(vecPointsT);

					cv::Mat vecPoints_vis(cv::Point3d(x_vis + x00, y_vis + y00, 1.0));
					Mat vecPoints_visT = vecPoints_vis.t();
					phosPoints_vis.push_back(vecPoints_visT);

				}

			}
		}
	}



	phosphenesPos = phosPoints_ray.t();
	phosphenesPos_vis = phosPoints_vis.t();


	Mat KDouble = Mat_<double>(3, 3);
	K.convertTo(KDouble, CV_64FC1);
	XiCam = cv::Mat_<double>::zeros(6, phosphenesPos.cols);
	cv::Mat v = KDouble.inv() * phosphenesPos;
	v.copyTo(XiCam.rowRange(0, 3));

}

static void genPhoshenSprite(Mat& sprite, double sigma)
{
	double y0_sprite, x0_sprite;
	y0_sprite = (sprite.rows - 1) / 2;
	x0_sprite = (sprite.cols - 1) / 2;

	for (int i = 0; i < sprite.rows; i++)
		for (int j = 0; j < sprite.cols; j++)
		{
			double d2 = pow(j - x0_sprite, 2) + pow(i - y0_sprite, 2);
			double P_x_c = exp(-d2 / 2 / pow(sigma, 2));

			unsigned char grayLevel = (unsigned char)round(P_x_c * 255);
			sprite.at<cv::Vec3b>(i, j) = cv::Vec3b(grayLevel, grayLevel, grayLevel);
		}

}

static void plotSprite(cv::Mat& img, cv::Mat& sprite, int xIni, int yIni)
{
	for (int i = 0; i < sprite.rows; i++)
	{
		if (yIni + i >= 0 && yIni + i < img.rows)
		{
			for (int j = 0; j < sprite.cols; j++)
			{
				if (xIni + j >= 0 && xIni + j < img.cols)
				{
					img.at<cv::Vec3b>(yIni + i, xIni + j) += sprite.at<cv::Vec3b>(i, j);
					//img.at<cv::Vec3b>(yIni + i, xIni + j) = sprite.at<cv::Vec3b>(i, j);
				}
			}
		}
	}
}

static void visualize(std::vector<int> phospheneFlag, cv::Mat& img, Mat phosphenesPos_vis, std::vector<cv::Mat> sprites, Mat sprite)
{
	for (int k = 0; k < phospheneFlag.size(); k++)
	{
		int iIndex = (int)round(phosphenesPos_vis.at<double>(1, k));
		int jIndex = (int)round(phosphenesPos_vis.at<double>(0, k));
		if (phospheneFlag[k] >= 1)
		{
			plotSprite(img, sprites[phospheneFlag[k]], jIndex - (sprite.cols - 1) / 2, iIndex - (sprite.rows - 1) / 2);
		}
	}
}

std::vector<int> TemporalModel(cv::Mat ampVector, cv::Mat caso, int nPhosphenes, int N_levels, double maxVal, double minVal, cv::Mat LUT)
{
	//creaci�n del vector de salida vac�o
	//cv::Mat matriz_imagen = cv::Mat_<double>::zeros(1600, 1);
	std::vector<int> vector_salida(nPhosphenes);

	double pixelPointScale = (N_levels / (maxVal - minVal));

	//std::cout << ampVector << std::endl;

	for (int n_electrodos = 0; n_electrodos < nPhosphenes; n_electrodos++) { //se emplea variable global(N_fos)

		for (int i = 0; i < 9; i++) {
			caso.at<unsigned char>(n_electrodos, i) = caso.at<unsigned char>(n_electrodos, i + 1);
		}
		caso.at<unsigned char>(n_electrodos, 9) = ampVector.at<unsigned char>(0, n_electrodos);

		unsigned char binario = 0;
		for (int i = 0; i < 9; i++) {
			//binario += (unsigned int)caso.at<unsigned char>(n_electrodos, i) << (9 - i);
			binario += (unsigned char)((double)caso.at<unsigned char>(n_electrodos, i) * std::pow(3, 9 - i));
		}
		if (caso.at<unsigned char>(n_electrodos, 9) == 2) {
			binario += 2;
		}
		else if (caso.at<unsigned char>(n_electrodos, 9) == 1) {
			binario += 1;
		}

		//unsigned int binario2 = 0;
		//for (int i = 0; i <= 9; i++) {
			//binario += (unsigned int)caso.at<unsigned char>(n_electrodos, i) << (9 - i);
			//binario2 += (unsigned int)((double)caso.at<unsigned char>(n_electrodos, i) * std::pow(3, 9 - i));
		//}
		//std::cout << binario << std::endl;
		/*std::cout << binario2 << std::endl;
		std::cout << " " << std::endl;*/

		// matriz_imagen.at<double>(n_electrodos, 0) = LUT.at<double>(binario, 0);
		vector_salida[n_electrodos] = (int)((LUT.at<double>(binario, 0) - minVal) * pixelPointScale);
		//std::cout << "n_electrodos: " << n_electrodos << std::endl;
	}

	// Perform manual normalization
	/*matriz_imagen = (matriz_imagen - minVal) * (7.0 / (maxVal - minVal));
	matriz_imagen.convertTo(matriz_imagen, CV_8U);
	vector_salida = matriz_imagen;*/

	return vector_salida;
}
// ------------------------END FUNCTIONS TO PHOSPHENES------------------------------

bool CMainApplication::ImageIntoPhosohenes() {
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

		cv::Mat srcbn8normalRight = cv::Mat::zeros(srcbn1Right.size(), srcbn1Right.type());

		for (int i = 0; i < srcbn1Right.rows; ++i) {
			for (int j = 0; j < srcbn1Right.cols; ++j) {
				srcbn8normalRight.at<unsigned char>(i, j) = srcbn1Right.at<unsigned char>(i, j) / 32;
			}
		}

		//Converting into phosphenes
		bool sleepFlag = true;

		plotImg.setTo(Vec3b(0, 0, 0));
		
		if (TemporalMode) {

			for (int i = 0; i < XiCam.cols; ++i) {
	
				float k = phosphenesPos.at<double>(0, i);
				float w = phosphenesPos.at<double>(1, i);
	
				if ((k >= 0) && (k < srcbn8normalRight.rows) && (w >= 0) && (w < srcbn8normalRight.cols) && (srcbn8normalRight.at<unsigned char>(k, w) > grayThreshold2)) {
					ampVectorRight.at<unsigned char>(0, i) = 2;
				}
				else if ((k >= 0) && (k < srcbn8normalRight.rows) && (w >= 0) && (w < srcbn8normalRight.cols) && (srcbn8normalRight.at<unsigned char>(k, w) > grayThreshold1)) {
					ampVectorRight.at<unsigned char>(0, i) = 1;
				}
				else {
					ampVectorRight.at<unsigned char>(0, i) = 0;
				}
	
			}
			std::vector<int> phospheneFlagRight(phosphenesPos.cols);
			phospheneFlagRight = TemporalModel(ampVectorRight, casoRight, phospheneFlagRight.size(), N_levels, maxVal, minVal, LUT);
			visualize(phospheneFlagRight, plotImg, phosphenesPos_vis, sprites, sprite);
		}
		else {
			for (int i = 0; i < XiCam.cols; ++i) {

				float k = phosphenesPos.at<double>(0, i);
				float w = phosphenesPos.at<double>(1, i);


				ampVectorRight.at<unsigned char>(0, i) = srcbn8normalRight.at<unsigned char>(k, w);

			}
			visualize(ampVectorRight, plotImg, phosphenesPos_vis, sprites, sprite);
		}
		// Pushing away calibrated image to have perspective in VR device. x is the calibration factor
		img_ph = cv::Mat::zeros(cv::Size(plotImg.rows * x_right, plotImg.cols * x_right), 16);
		uint32_t m = (plotImg.rows * (x_right - 1) / 2); //distance between left screen limit and beggining of the picture
		uint32_t n = (plotImg.cols * (x_right - 1) / 2); //distance between top screen limit and beggining of the picture
		for (uint32_t a = 0; a < plotImg.cols; a++) {
			for (uint32_t b = 0; b < plotImg.rows; b++) {
				img_ph.at<cv::Vec3b>(a + n, b + m) = plotImg.at<cv::Vec3b>(a, b);
			}
		}
		
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

		if (sleepFlag)
			std::this_thread::sleep_for(std::chrono::milliseconds(10 - static_cast<int>(time_span.count())));

        return true;
	
}

static GLuint matToTexture(const cv::Mat& mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter, bool iniciogl, GLuint textureID) {
	// Generate a number for our textureID's unique handle

	// Bind to our texture handle
	glBindTexture(GL_TEXTURE_2D, textureID);

	GLenum inputColourFormat = GL_BGR;
	if (mat.channels() == 1)
	{
		inputColourFormat = GL_LUMINANCE;
	}


	// Catch silly-mistake texture interpolation method for magnification
	if (magFilter == GL_LINEAR_MIPMAP_LINEAR ||
		magFilter == GL_LINEAR_MIPMAP_NEAREST ||
		magFilter == GL_NEAREST_MIPMAP_LINEAR ||
		magFilter == GL_NEAREST_MIPMAP_NEAREST)
	{
		std::cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << std::endl;
		magFilter = GL_LINEAR;
	}

	// Set texture interpolation methods for minification and magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

	// Set incoming texture format to:
	// GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
	// GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
	// Work out other mappings as required ( there's a list in comments in main() )


	//use fast 4-byte alignment (default anyway) if possible 
	glPixelStorei(GL_UNPACK_ALIGNMENT, (mat.step & 3) ? 1 : 4);
	//set length of one complete row in data (doesn't need to equal image.cols) 
	glPixelStorei(GL_UNPACK_ROW_LENGTH, mat.step / mat.elemSize());


	if (iniciogl)
	{
	// Create the texture
		glTexImage2D(GL_TEXTURE_2D,     // Type of texture
			0,                 // Pyramid level (for mip-mapping) - 0 is the top level
			GL_RGBA8,            // Internal colour format to convert to
			mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
			mat.rows,          // Image height i.e. 480 for Kinect in standard mode
			0,                 // Border width in pixels (can either be 1 or 0)
			inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
			GL_UNSIGNED_BYTE,  // Image data type
			mat.ptr());        // The actual image data itself
	}
	else
	{
		glTexSubImage2D(GL_TEXTURE_2D,     // Type of texture
			0,                 // Pyramid level (for mip-mapping) - 0 is the top level
			0, 0,           // xy offset
			mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
			mat.rows,          // Image height i.e. 480 for Kinect in standard mode
			inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
			GL_UNSIGNED_BYTE,  // Image data type
			mat.ptr());        // The actual image data itself
	}

	return textureID;
}

//-----------------------------------------------------------------------------
// Purpose: Outputs a set of optional arguments to debugging output, using
//          the printf format setting specified in fmt*.
//-----------------------------------------------------------------------------
void dprintf(const char* fmt, ...)
{
	va_list args;
	char buffer[2048];

	va_start(args, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, args);
	va_end(args);

	if (g_bPrintf)
		printf("%s", buffer);

	//OutputDebugStringA(buffer);
	std::cout << buffer << std::endl;
}

//-----------------------------------------------------------------------------
// Purpose:
//-----------------------------------------------------------------------------
void CMainApplication::RenderFrame()
{

	if (m_cam.isOpened())
    {
        cv::Mat frame;
        if (m_cam.read(frame))
        {
            for(size_t i=0; i<roi.size(); i++)
            {
                cv::Mat img = frame(roi[i]);
				srcbn1Right = SingleViewToPhosphenes(img, 128, 48, 12 * 32, 12 * 32);
                ImageIntoPhosohenes();
				
				// Muestra en una ventana
				cv::Mat imgshow = img_ph;
				cv::rotate(imgshow, imgshow, cv::ROTATE_90_CLOCKWISE);
				cv::namedWindow(window_name[i], cv::WINDOW_NORMAL);
				cv::imshow(window_name[i], imgshow.clone());

                // Images have to be rotate horizontally because it appears a mirror effect
                //cv::rotate(img_ph, img_ph, cv::ROTATE_90_COUNTERCLOCKWISE);
				cv::flip(img_ph, img_ph, -1);
                // Converting into textures and submitting (into VR screens and in a window in the computer screen)
                GLuint& texture = (i == 0) ? image_tex_Left : image_tex_Right; // Select the texture
                texture = matToTexture(img_ph, GL_LINEAR, GL_LINEAR, GL_CLAMP, iniciogl, texture);
                
                vr::Texture_t eyeTexture = { (void*)(uintptr_t)texture, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
                vr::EVREye eye = (i == 0) ? vr::Eye_Left : vr::Eye_Right; // Asign the correct eye
                vr::VRCompositor()->Submit(eye, &eyeTexture);
                
                glFinish();
            }
            
            iniciogl = false;
            cv::waitKey(1);
            UpdateHMDMatrixPose();

        }
    }
}


bool CMainApplication::Initialitation() {
	std::cout << "\nStarting OpenVR...\n";
	iniciogl = true;
	glGenTextures(1, &image_tex_Left);
	glGenTextures(1, &image_tex_Right);


	bool stereo = true; // Estéreo activado por defecto
	double width  = m_cam.get(cv::CAP_PROP_FRAME_WIDTH);
	double height = m_cam.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "Default WxH : " << width << " x " << height << std::endl;

	if(height > width) // this means it is already in stereo mode
	{
		height = height/2;
	}

	if (stereo)
	{
		std::cout << "Height before : " << height << std::endl;
		m_cam.set(cv::CAP_PROP_FRAME_HEIGHT, height*2);
		std::cout << "Height : " << height << std::endl;
		roi = {cv::Rect(0, height, width, height), cv::Rect(0, 0, width, height)};
		std::cout << height << "&" << width<< std::endl;
		window_name = {"Left", "Right"};
	}
	else
	{
		roi = {cv::Rect(0, 0, width, height)};
		window_name = {"Default"};
	}
	

	std::string txtFileName = "prothesis_simulator.cfg"; // Reemplaza "archivo.csv" con la ruta de tu archivo CSV

	// Abrir el archivo CSV
	std::ifstream file(txtFileName);
	if (!file.is_open()) {
		cout << "Could not open the file - '" << endl;
		exit(EXIT_FAILURE);
	}
	std::vector<double> datos; // Matriz columna LUT;
	std::string linea;
	// Leer y procesar cada l�nea del archivo CSV
	while (std::getline(file, linea)) {
		std::istringstream ss(linea);
		std::string celda;

		while (std::getline(ss, celda, 'n')) {
			double valor = std::stod(celda); // Convertir el valor a double
			datos.push_back(valor);
		}
	}
	file.close();

	TemporalMode = datos[0];
	x_right = datos[1];
	N_levels = datos[3];

    std::cout<<N_levels<<std::endl;

	plotImg = cv::Mat::zeros(cv::Size(384, 384), 16);

	X0 = plotImg.cols / 2;
	fx = 196;
	fy = 189;

	Y0 = plotImg.rows / 2;
	K = (Mat_<double>(3, 3) << fx, 0, X0, 0, fy, Y0, 0, 0, 1);
	f = (fx + fy) / 2;


	// Phosphene Matrix configuration
	deltaX = computeDeltaFromNumberOfPhosphenes(plotImg.rows, plotImg.cols, N_fos);
	deltaY = deltaX;
	rExt = sqrt(X0 * X0 + Y0 * Y0);
	genPhosphenesPos(plotImg.rows, plotImg.cols, rExt, X0, Y0, deltaX, deltaY, grid_mode, noise_stddev, phosphenesPos, phosphenesPos_vis, XiCam, K);
	// Phosphene sprite configuration
	size_sprite = deltaX * 2;
	reference_sprite = cv::Mat_<cv::Vec3b>::zeros(32, 32);
	reference_sprite.setTo(cv::Scalar(255, 255, 255));
	genPhoshenSprite(reference_sprite, 3.5); 
	resize(reference_sprite, sprite, cv::Size(size_sprite, size_sprite));
	for (size_t i = 0; i <= N_levels; i++)
	{
		cv::Mat sprite_aux = sprite.clone();
		double dim_factor = double(i) / double(N_levels);
		sprites.push_back(sprite_aux * dim_factor);
	}

	ampVectorRight = cv::Mat::zeros(1, phosphenesPos.cols, CV_8U);
	
	if (TemporalMode) {
		//Tabulation variable for the temporal model
		//LUT = (cv::Mat_<double>(1024, 1) << 0.0000000000, 0.0000004019, 38.0088969959, 38.0088973978, 71.1527607173, 71.1527611192, 109.1616576614, 109.1616580633, 74.7457382118, 74.7457386138, 112.7546351225, 112.7546355244, 145.8984988325, 145.8984992344, 183.9073956914, 183.9073960933, 62.0037020973, 62.0037024992, 100.0125989864, 100.0125993884, 133.1564626554, 133.1564630573, 171.1653594928, 171.1653598947, 136.7494402077, 136.7494406096, 174.7583370117, 174.7583374136, 207.9022006692, 207.9022010711, 245.9110974214, 245.9110978233, 45.1947513066, 45.1947517085, 83.2036481819, 83.2036485838, 116.3475118245, 116.3475122264, 154.3564086481, 154.3564090500, 119.9404893514, 119.9404897533, 157.9493861415, 157.9493865434, 191.0932497727, 191.0932501746, 229.1021465111, 229.1021469130, 107.1984533198, 107.1984537217, 145.2073500884, 145.2073504903, 178.3512136786, 178.3512140805, 216.3601103954, 216.3601107973, 181.9441912631, 181.9441916650, 219.9530879465, 219.9530883484, 253.0969515253, 253.0969519272, 291.1058481569, 291.1058485588, 30.3563236001, 30.3563240020, 68.3652204666, 68.3652208685, 101.5090840923, 101.5090844942, 139.5179809070, 139.5179813089, 105.1020616027, 105.1020620046, 143.1109583840, 143.1109587859, 176.2548219982, 176.2548224001, 214.2637187277, 214.2637191296, 92.3600255589, 92.3600259608, 130.3689223186, 130.3689227205, 163.5127858918, 163.5127862938, 201.5216825998, 201.5216830018, 167.1057634601, 167.1057638620, 205.1146601346, 205.1146605365, 238.2585236964, 238.2585240983, 276.2674203192, 276.2674207211, 75.5510748454, 75.5510752473, 113.5599715913, 113.5599719933, 146.7038351382, 146.7038355401, 184.7127318324, 184.7127322343, 150.2968126809, 150.2968130829, 188.3057093416, 188.3057097436, 221.4495728771, 221.4495732790, 259.4584694861, 259.4584698880, 137.5547767201, 137.5547771220, 175.5636733593, 175.5636737612, 208.7075368537, 208.7075372556, 246.7164334412, 246.7164338431, 212.3005144542, 212.3005148561, 250.3094110081, 250.3094114101, 283.4532744912, 283.4532748931, 321.4621709934, 321.4621713953, 19.2713397702, 19.2713401721, 57.2802366309, 57.2802370328, 90.4241002457, 90.4241006476, 128.4329970547, 128.4329974567, 94.0170777457, 94.0170781476, 132.0259745212, 132.0259749231, 165.1698381246, 165.1698385265, 203.1787348484, 203.1787352503, 81.2750416940, 81.2750420959, 119.2839384480, 119.2839388499, 152.4278020103, 152.4278024123, 190.4366987126, 190.4366991146, 156.0207795680, 156.0207799699, 194.0296762368, 194.0296766388, 227.1735397878, 227.1735401897, 265.1824364049, 265.1824368068, 64.4660909758, 64.4660913777, 102.4749877160, 102.4749881179, 135.6188512520, 135.6188516539, 173.6277479405, 173.6277483424, 139.2118287842, 139.2118291861, 177.2207254392, 177.2207258411, 210.3645889638, 210.3645893657, 248.3734855671, 248.3734859690, 126.4697928155, 126.4697932174, 164.4786894490, 164.4786898509, 197.6225529325, 197.6225533344, 235.6314495143, 235.6314499162, 201.2155305225, 201.2155309244, 239.2244270707, 239.2244274727, 272.3682905429, 272.3682909448, 310.3771870394, 310.3771874413, 49.6276633292, 49.6276637311, 87.6365600605, 87.6365604624, 120.7804235795, 120.7804239815, 158.7893202591, 158.7893206611, 124.3734010954, 124.3734014973, 162.3822977415, 162.3822981434, 195.5261612492, 195.5261616511, 233.5350578436, 233.5350582455, 111.6313651144, 111.6313655163, 149.6402617390, 149.6402621409, 182.7841252056, 182.7841256075, 220.7930217785, 220.7930221804, 186.3771027792, 186.3771031811, 224.3859993186, 224.3859997205, 257.5298627738, 257.5298631757, 295.5387592615, 295.5387596634, 94.8224144735, 94.8224148754, 132.8313110843, 132.8313114862, 165.9751745245, 165.9751749264, 203.9840710836, 203.9840714855, 169.5681520727, 169.5681524746, 207.5770485982, 207.5770490001, 240.7209120271, 240.7209124290, 278.7298085009, 278.7298089028, 156.8261161746, 156.8261165765, 194.8350126787, 194.8350130806, 227.9788760665, 227.9788764684, 265.9877725188, 265.9877729207, 231.5718536724, 231.5718540743, 269.5807500912, 269.5807504931, 302.7246134676, 302.7246138695, 340.7335098347, 340.7335102366, 11.7394163468, 11.7394167487, 49.7483132039, 49.7483136058, 82.8921768117, 82.8921772136, 120.9010736171, 120.9010740190, 86.4851543049, 86.4851547068, 124.4940510768, 124.4940514787, 157.6379146732, 157.6379150751, 195.6468113933, 195.6468117952, 73.7431182481, 73.7431186500, 111.7520149985, 111.7520154004, 144.8958785538, 144.8958789558, 182.9047752525, 182.9047756544, 148.4888561048, 148.4888565067, 186.4977527699, 186.4977531719, 219.6416163139, 219.6416167158, 257.6505129273, 257.6505133292, 56.9341675269, 56.9341679289, 94.9430642635, 94.9430646654, 128.0869277925, 128.0869281944, 166.0958244773, 166.0958248792, 131.6799053180, 131.6799057199, 169.6888019693, 169.6888023712, 202.8326654869, 202.8326658888, 240.8415620865, 240.8415624884, 118.9378693442, 118.9378697461, 156.9467659740, 156.9467663759, 190.0906294505, 190.0906298524, 228.0995260286, 228.0995264305, 193.6836070337, 193.6836074356, 231.6925035783, 231.6925039803, 264.8363670435, 264.8363674454, 302.8452635364, 302.8452639383, 42.0957398791, 42.0957402810, 80.1046366068, 80.1046370087, 113.2485001189, 113.2485005208, 151.2573967948, 151.2573971967, 116.8414776280, 116.8414780299, 154.8503742704, 154.8503746723, 187.9942377711, 187.9942381730, 226.0031343618, 226.0031347637, 104.0994416419, 104.0994420438, 142.1083382628, 142.1083386648, 175.2522017225, 175.2522021244, 213.2610982917, 213.2610986936, 178.8451792893, 178.8451796912, 216.8540758250, 216.8540762270, 249.9979392733, 249.9979396752, 288.0068357573, 288.0068361592, 87.2904909980, 87.2904913999, 125.2993876051, 125.2993880070, 158.4432510384, 158.4432514403, 196.4521475938, 196.4521479957, 162.0362285797, 162.0362289816, 200.0451251017, 200.0451255036, 233.1889885235, 233.1889889254, 271.1978849937, 271.1978853956, 149.2941926766, 149.2941930786, 187.3030891770, 187.3030895790, 220.4469525579, 220.4469529598, 258.4558490065, 258.4558494085, 224.0399301570, 224.0399305589, 262.0488265722, 262.0488269741, 295.1926899416, 295.1926903435, 333.2015863050, 333.2015867070, 31.0107560909, 31.0107564928, 69.0196528128, 69.0196532147, 102.1635163140, 102.1635167159, 140.1724129843, 140.1724133862, 105.7564938126, 105.7564942145, 143.7653904494, 143.7653908513, 176.9092539391, 176.9092543411, 214.9181505242, 214.9181509261, 93.0144578187, 93.0144582206, 131.0233544339, 131.0233548358, 164.1672178826, 164.1672182846, 202.1761144462, 202.1761148481, 167.7601954390, 167.7601958409, 205.7690919690, 205.7690923709, 238.9129554063, 238.9129558083, 276.9218518846, 276.9218522866, 76.2055071700, 76.2055075719, 114.2144037715, 114.2144041734, 147.3582671938, 147.3582675958, 185.3671637435, 185.3671641455, 150.9512447247, 150.9512451266, 188.9601412409, 188.9601416428, 222.1040046519, 222.1040050538, 260.1129011164, 260.1129015183, 138.2092088137, 138.2092092156, 176.2181053084, 176.2181057103, 209.3619686784, 209.3619690803, 247.3708651213, 247.3708655233, 212.9549462670, 212.9549466689, 250.9638426765, 250.9638430784, 284.1077060350, 284.1077064369, 322.1166023927, 322.1166027947, 61.3670795820, 61.3670799839, 99.3759761746, 99.3759765765, 132.5198395800, 132.5198399819, 170.5287361209, 170.5287365228, 136.1128170945, 136.1128174964, 174.1217136018, 174.1217140038, 207.2655769959, 207.2655773978, 245.2744734515, 245.2744738534, 123.3707811713, 123.3707815732, 161.3796776571, 161.3796780590, 194.5235410101, 194.5235414120, 232.5324374442, 232.5324378461, 198.1165185823, 198.1165189842, 236.1254149830, 236.1254153849, 269.2692783245, 269.2692787265, 307.2781746734, 307.2781750753, 106.5618305999, 106.5618310018, 144.5707270719, 144.5707274738, 177.7145903986, 177.7145908005, 215.7234868188, 215.7234872208, 181.3075679453, 181.3075683472, 219.3164643321, 219.3164647340, 252.4603276473, 252.4603280493, 290.4692239824, 290.4692243843, 168.5655321050, 168.5655325070, 206.5744284703, 206.5744288723, 239.7182917445, 239.7182921465, 277.7271880581, 277.7271884600, 243.3112693490, 243.3112697509, 281.3201656291, 281.3201660310, 314.4640288919, 314.4640292938, 352.4729251202, 352.4729255222, 6.9293093822, 6.9293097841, 44.9382062369, 44.9382066388, 78.0820698402, 78.0820702421, 116.0909666432, 116.0909670452, 81.6750473291, 81.6750477310, 119.6839440986, 119.6839445005, 152.8278076905, 152.8278080924, 190.8367044083, 190.8367048102, 68.9330112691, 68.9330116710, 106.9419080171, 106.9419084190, 140.0857715680, 140.0857719699, 178.0946682642, 178.0946686661, 143.6787491146, 143.6787495165, 181.6876457774, 181.6876461793, 214.8315093168, 214.8315097187, 252.8404059279, 252.8404063298, 52.1240605459, 52.1240609479, 90.1329572801, 90.1329576821, 123.2768208047, 123.2768212066, 161.2857174871, 161.2857178890, 126.8697983258, 126.8697987277, 164.8786949748, 164.8786953767, 198.0225584879, 198.0225588898, 236.0314550851, 236.0314554870, 114.1277623487, 114.1277627506, 152.1366589762, 152.1366593781, 185.2805224483, 185.2805228502, 223.2894190240, 223.2894194259, 188.8735000271, 188.8735004290, 226.8823965694, 226.8823969713, 260.0262600300, 260.0262604320, 298.0351565206, 298.0351569225, 37.2856328974, 37.2856332993, 75.2945296227, 75.2945300246, 108.4383931303, 108.4383935322, 146.4472898039, 146.4472902058, 112.0313706351, 112.0313710370, 150.0402672752, 150.0402676771, 183.1841307713, 183.1841311732, 221.1930273597, 221.1930277616, 99.2893346457, 99.2893350476, 137.2982312643, 137.2982316662, 170.4420947195, 170.4420951214, 208.4509912863, 208.4509916882, 174.0350722820, 174.0350726839, 212.0439688153, 212.0439692173, 245.1878322591, 245.1878326610, 283.1967287407, 283.1967291426, 82.4803839999, 82.4803844018, 120.4892806046, 120.4892810066, 153.6331440334, 153.6331444353, 191.6420405865, 191.6420409884, 157.2261215704, 157.2261219724, 195.2350180900, 195.2350184919, 228.3788815074, 228.3788819093, 266.3877779752, 266.3877783771, 144.4840856641, 144.4840860660, 182.4929821622, 182.4929825641, 215.6368455385, 215.6368459404, 253.6457419848, 253.6457423867, 219.2298231332, 219.2298235352, 257.2387195461, 257.2387199480, 290.3825829110, 290.3825833129, 328.3914792721, 328.3914796740, 26.2006491093, 26.2006495112, 64.2095458289, 64.2095462309, 97.3534093256, 97.3534097276, 135.3623059935, 135.3623063954, 100.9463868199, 100.9463872218, 138.9552834543, 138.9552838562, 172.0991469396, 172.0991473415, 210.1080435223, 210.1080439242, 88.2043508227, 88.2043512246, 126.2132474356, 126.2132478375, 159.3571108798, 159.3571112818, 197.3660074410, 197.3660078429, 162.9500884318, 162.9500888337, 200.9589849595, 200.9589853614, 234.1028483923, 234.1028487943, 272.1117448683, 272.1117452702, 71.3954001721, 71.3954005740, 109.4042967712, 109.4042971731, 142.5481601891, 142.5481605910, 180.5570567364, 180.5570571384, 146.1411377156, 146.1411381175, 184.1500342295, 184.1500346314, 217.2938976360, 217.2938980379, 255.3027940981, 255.3027945000, 133.3991018014, 133.3991022033, 171.4079982937, 171.4079986957, 204.5518616592, 204.5518620611, 242.5607580998, 242.5607585017, 208.1448392434, 208.1448396454, 246.1537356506, 246.1537360525, 279.2975990046, 279.2975994065, 317.3064953600, 317.3064957619, 56.5569725834, 56.5569729853, 94.5658691736, 94.5658695755, 127.7097325745, 127.7097329764, 165.7186291130, 165.7186295149, 131.3027100847, 131.3027104866, 169.3116065897, 169.3116069916, 202.4554699792, 202.4554703811, 240.4643664325, 240.4643668344, 118.5606741582, 118.5606745601, 156.5695706417, 156.5695710436, 189.7134339902, 189.7134343921, 227.7223304219, 227.7223308238, 193.3064115581, 193.3064119600, 231.3153079563, 231.3153083582, 264.4591712934, 264.4591716954, 302.4680676400, 302.4680680419, 101.7517235849, 101.7517239868, 139.7606200545, 139.7606204565, 172.9044833767, 172.9044837786, 210.9133797946, 210.9133801965, 176.4974609191, 176.4974613210, 214.5063573036, 214.5063577055, 247.6502206143, 247.6502210162, 285.6591169470, 285.6591173489, 163.7554250756, 163.7554254775, 201.7643214385, 201.7643218404, 234.9081847082, 234.9081851102, 272.9170810195, 272.9170814214, 238.5011623084, 238.5011627103, 276.5100585861, 276.5100589880, 309.6539218444, 309.6539222463, 347.6628180704, 347.6628184723, 18.6687257131, 18.6687261150, 56.6776224290, 56.6776228309, 89.8214859187, 89.8214863207, 127.8303825830, 127.8303829849, 93.4144634062, 93.4144638081, 131.4233600370, 131.4233604389, 164.5672235153, 164.5672239172, 202.5761200943, 202.5761204962, 80.6724274040, 80.6724278059, 118.6813240132, 118.6813244151, 151.8251874505, 151.8251878524, 189.8340840080, 189.8340844099, 155.4181649957, 155.4181653976, 193.4270615197, 193.4270619216, 226.5709249456, 226.5709253475, 264.5798214179, 264.5798218198, 63.8634767504, 63.8634771523, 101.8723733458, 101.8723737477, 135.0162367567, 135.0162371586, 173.0251333004, 173.0251337023, 138.6092142765, 138.6092146784, 176.6181107867, 176.6181111886, 209.7619741862, 209.7619745881, 247.7708706446, 247.7708710465, 125.8671783572, 125.8671787591, 163.8760748459, 163.8760752478, 197.0199382043, 197.0199386062, 235.0288346413, 235.0288350432, 200.6129157818, 200.6129161837, 238.6218121853, 238.6218125872, 271.7656755324, 271.7656759343, 309.7745718841, 309.7745722860, 49.0250491605, 49.0250495624, 87.0339457470, 87.0339461489, 120.1778091410, 120.1778095429, 158.1867056758, 158.1867060777, 123.7707866444, 123.7707870463, 161.7796831457, 161.7796835476, 194.9235465283, 194.9235469302, 232.9324429779, 232.9324433798, 111.0287507128, 111.0287511147, 149.0376471926, 149.0376475946, 182.1815105342, 182.1815109361, 220.1904069623, 220.1904073642, 185.7744880953, 185.7744884972, 223.7833844899, 223.7833848918, 256.9272478200, 256.9272482219, 294.9361441629, 294.9361445648, 94.2198001365, 94.2198005384, 132.2286966025, 132.2286970044, 165.3725599177, 165.3725603196, 203.3814563319, 203.3814567338, 168.9655374533, 168.9655378552, 206.9744338341, 206.9744342360, 240.1182971379, 240.1182975398, 278.1271934669, 278.1271938688, 156.2235016047, 156.2235020067, 194.2323979640, 194.2323983659, 227.3762612267, 227.3762616287, 265.3851575343, 265.3851579362, 230.9692388201, 230.9692392221, 268.9781350942, 268.9781354961, 302.1219983455, 302.1219987474, 340.1308945679, 340.1308949698, 37.9400654141, 37.9400658160, 75.9489619949, 75.9489623968, 109.0928253780, 109.0928257799, 147.1017219071, 147.1017223090, 112.6858028709, 112.6858032728, 150.6946993665, 150.6946997684, 183.8385627382, 183.8385631401, 221.8474591821, 221.8474595840, 99.9437669315, 99.9437673334, 137.9526634056, 137.9526638075, 171.0965267362, 171.0965271381, 209.1054231586, 209.1054235605, 174.6895042868, 174.6895046888, 212.6984006757, 212.6984010777, 245.8422639950, 245.8422643969, 283.8511603322, 283.8511607341, 83.1348163504, 83.1348167523, 121.1437128107, 121.1437132126, 154.2875761150, 154.2875765169, 192.2964725236, 192.2964729255, 157.8805536402, 157.8805540421, 195.8894500152, 195.8894504172, 229.0333133081, 229.0333137100, 267.0422096315, 267.0422100334, 145.1385177837, 145.1385181856, 183.1474141373, 183.1474145392, 216.2912773891, 216.2912777910, 254.3001736910, 254.3001740929, 219.8842549720, 219.8842553739, 257.8931512404, 257.8931516423, 291.0370144808, 291.0370148827, 329.0459106974, 329.0459110994, 68.2963888203, 68.2963892222, 106.3052852717, 106.3052856737, 139.4491485591, 139.4491489610, 177.4580449588, 177.4580453607, 143.0421260679, 143.0421264698, 181.0510224341, 181.0510228360, 214.1948857100, 214.1948861119, 252.2037820245, 252.2037824264, 130.3000901992, 130.3000906011, 168.3089865439, 168.3089869458, 201.4528497788, 201.4528501807, 239.4617460717, 239.4617464736, 205.0458273453, 205.0458277472, 243.0547236048, 243.0547240067, 276.1985868283, 276.1985872302, 314.2074830360, 314.2074834379, 113.4911396954, 113.4911400973, 151.5000360263, 151.5000364282, 184.6438992348, 184.6438996367, 222.6527955140, 222.6527959159, 188.2368767759, 188.2368771778, 226.2457730215, 226.2457734234, 259.3896362187, 259.3896366206, 297.3985324126, 297.3985328145, 175.4948409901, 175.4948413920, 213.5037372143, 213.5037376162, 246.6476003704, 246.6476007723, 284.6564965428, 284.6564969447, 250.2405779692, 250.2405783711, 288.2494741081, 288.2494745100, 321.3933372528, 321.3933376547, 359.4022333400, 359.4022337419);

		std::string csvFileName = "tabulacion3.csv"; // Reemplaza "archivo.csv" con la ruta de tu archivo CSV
		//int columnIndex = 0; // �ndice de la columna a leer (comenzando desde 0)

		std::vector<double> columnVector; // Matriz columna LUT;

		// Abrir el archivo CSV
		std::ifstream file2(csvFileName);

		if (!file2.is_open()) {
			cout << "Could not open the file - '" << endl;
			exit(EXIT_FAILURE);
		}
		std::string line;

		// Leer y procesar cada l�nea del archivo CSV
		while (std::getline(file2, line)) {
			std::istringstream ss(line);
			std::string cell;

			while (std::getline(ss, cell, ',')) {
				double value = std::stod(cell); // Convertir el valor a double
				columnVector.push_back(value);
			}
		}

		// Cerrar el archivo CSV
		file2.close();

		// Crear una matriz cv::Mat a partir del vector
		LUT = cv::Mat(columnVector.size(), 1, CV_64F);

		//Copiar los valores en la nueva variable
		for (int i = 0; i < columnVector.size(); i++) {
			LUT.at<double>(i, 0) = columnVector[i];
		}

		cv::minMaxLoc(LUT, &minVal, &maxVal);
	}

	//creation of the temporal model variables
	casoRight = cv::Mat::zeros(phosphenesPos.cols, 10, CV_8U);
	casoLeft = cv::Mat::zeros(phosphenesPos.cols, 10, CV_8U);
	grayThreshold1 = 1;
	grayThreshold2 = 6;

    return true;
}

//-----------------------------------------------------------------------------
// Purpose: Constructor
//-----------------------------------------------------------------------------
CMainApplication::CMainApplication( int argc, char *argv[] )
	: m_pCompanionWindow(NULL)
	, m_pContext(NULL)
	, m_nCompanionWindowWidth( 640 )
	, m_nCompanionWindowHeight( 320 )
	, m_unSceneProgramID( 0 )
	, m_unCompanionWindowProgramID( 0 )
	, m_unControllerTransformProgramID( 0 )
	, m_unRenderModelProgramID( 0 )
	, m_pHMD( NULL )
	, m_bDebugOpenGL( false )
	, m_bVerbose( false )
	, m_bPerf( false )
	, m_bVblank( false )
	, m_bGlFinishHack( true )
	, m_unControllerVAO( 0 )
	, m_unSceneVAO( 0 )
	, m_iValidPoseCount( 0 )
	, m_iSceneVolumeInit( 20 )
	, m_strPoseClasses("")
	, m_bShowCubes( true )
	, m_cam(0)
{

	for( int i = 1; i < argc; i++ )
	{
		if( !strcasecmp( argv[i], "-gldebug" ) )
		{
			m_bDebugOpenGL = true;
		}
		else if( !strcasecmp( argv[i], "-verbose" ) )
		{
			m_bVerbose = true;
		}
		else if( !strcasecmp( argv[i], "-novblank" ) )
		{
			m_bVblank = false;
		}
		else if( !strcasecmp( argv[i], "-noglfinishhack" ) )
		{
			m_bGlFinishHack = false;
		}
		else if( !strcasecmp( argv[i], "-noprintf" ) )
		{
			g_bPrintf = false;
		}
		else if ( !strcasecmp( argv[i], "-cubevolume" ) && ( argc > i + 1 ) && ( *argv[ i + 1 ] != '-' ) )
		{
			m_iSceneVolumeInit = atoi( argv[ i + 1 ] );
			i++;
		}
	}
	// other initialization tasks are done in BInit
	memset(m_rDevClassChar, 0, sizeof(m_rDevClassChar));
};


//-----------------------------------------------------------------------------
// Purpose: Destructor
//-----------------------------------------------------------------------------
CMainApplication::~CMainApplication()
{
	// work is done in Shutdown
	dprintf( "Shutdown" );
}

//-----------------------------------------------------------------------------
// Purpose:
//-----------------------------------------------------------------------------
bool CMainApplication::BInit()
{
	if ( SDL_Init( SDL_INIT_VIDEO | SDL_INIT_TIMER ) < 0 )
	{
		printf("%s - SDL could not initialize! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	// Loading the SteamVR Runtime
	vr::EVRInitError eError = vr::VRInitError_None;
	m_pHMD = vr::VR_Init( &eError, vr::VRApplication_Scene );

	if (!m_pHMD)
	{
		throw std::runtime_error("Error : presenting frames when VR system handle is NULL");
	}

	if ( eError != vr::VRInitError_None )
	{
		m_pHMD = NULL;
		char buf[1024];
		snprintf( buf, sizeof( buf ), "Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription( eError ) );
		SDL_ShowSimpleMessageBox( SDL_MESSAGEBOX_ERROR, "VR_Init Failed", buf, NULL );
		return false;
	}

	int nWindowPosX = 700;
	int nWindowPosY = 100;
	Uint32 unWindowFlags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;

	SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 4 );
	SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 1 );
	SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE );

	SDL_GL_SetAttribute( SDL_GL_MULTISAMPLEBUFFERS, 0 );
	SDL_GL_SetAttribute( SDL_GL_MULTISAMPLESAMPLES, 0 );
	if( m_bDebugOpenGL )
		SDL_GL_SetAttribute( SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG );

	m_pCompanionWindow = SDL_CreateWindow( "hellovr", nWindowPosX, nWindowPosY, m_nCompanionWindowWidth, m_nCompanionWindowHeight, unWindowFlags );
	if (m_pCompanionWindow == NULL)
	{
		printf( "%s - Window could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError() );
		return false;
	}

	m_pContext = SDL_GL_CreateContext(m_pCompanionWindow);
	if (m_pContext == NULL)
	{
		printf( "%s - OpenGL context could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError() );
		return false;
	}
	
	glewExperimental = GL_TRUE;
	GLenum nGlewError = glewInit();

	return true;
}

//-----------------------------------------------------------------------------
// Purpose:
//-----------------------------------------------------------------------------
void CMainApplication::Shutdown()
{
	if( m_pHMD )
	{
		vr::VR_Shutdown();
		m_pHMD = NULL;
	}

	for( std::vector< CGLRenderModel * >::iterator i = m_vecRenderModels.begin(); i != m_vecRenderModels.end(); i++ )
	{
		delete (*i);
	}
	m_vecRenderModels.clear();
	
	if( m_pContext )
	{
		if( m_bDebugOpenGL )
		{
			glDebugMessageControl( GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_FALSE );
			glDebugMessageCallback(nullptr, nullptr);
		}
		glDeleteBuffers(1, &m_glSceneVertBuffer);

		if ( m_unSceneProgramID )
		{
			glDeleteProgram( m_unSceneProgramID );
		}
		if ( m_unControllerTransformProgramID )
		{
			glDeleteProgram( m_unControllerTransformProgramID );
		}
		if ( m_unRenderModelProgramID )
		{
			glDeleteProgram( m_unRenderModelProgramID );
		}
		if ( m_unCompanionWindowProgramID )
		{
			glDeleteProgram( m_unCompanionWindowProgramID );
		}

		glDeleteRenderbuffers( 1, &leftEyeDesc.m_nDepthBufferId );
		glDeleteTextures( 1, &leftEyeDesc.m_nRenderTextureId );
		glDeleteFramebuffers( 1, &leftEyeDesc.m_nRenderFramebufferId );
		glDeleteTextures( 1, &leftEyeDesc.m_nResolveTextureId );
		glDeleteFramebuffers( 1, &leftEyeDesc.m_nResolveFramebufferId );

		glDeleteRenderbuffers( 1, &rightEyeDesc.m_nDepthBufferId );
		glDeleteTextures( 1, &rightEyeDesc.m_nRenderTextureId );
		glDeleteFramebuffers( 1, &rightEyeDesc.m_nRenderFramebufferId );
		glDeleteTextures( 1, &rightEyeDesc.m_nResolveTextureId );
		glDeleteFramebuffers( 1, &rightEyeDesc.m_nResolveFramebufferId );

		if( m_unCompanionWindowVAO != 0 )
		{
			glDeleteVertexArrays( 1, &m_unCompanionWindowVAO );
		}
		if( m_unSceneVAO != 0 )
		{
			glDeleteVertexArrays( 1, &m_unSceneVAO );
		}
		if( m_unControllerVAO != 0 )
		{
			glDeleteVertexArrays( 1, &m_unControllerVAO );
		}
	}

	SDL_Quit();
}

//-----------------------------------------------------------------------------
// Purpose:
//-----------------------------------------------------------------------------
bool CMainApplication::HandleInput()
{
	SDL_Event sdlEvent;
	bool bRet = false;

	while ( SDL_PollEvent( &sdlEvent ) != 0 )
	{
		if ( sdlEvent.type == SDL_QUIT )
		{
			bRet = true;
		}
		else if ( sdlEvent.type == SDL_KEYDOWN )
		{
			if ( sdlEvent.key.keysym.sym == SDLK_ESCAPE 
			     || sdlEvent.key.keysym.sym == SDLK_q )
			{
				bRet = true;
			}
			if( sdlEvent.key.keysym.sym == SDLK_c )
			{
				m_bShowCubes = !m_bShowCubes;
			}
		}
	}

	return bRet;
}

//-----------------------------------------------------------------------------
// Purpose:
//-----------------------------------------------------------------------------
void CMainApplication::RunMainLoop()
{
	bool bQuit = false;
	Initialitation();

	while ( !bQuit )
	{
		bQuit = HandleInput();
		RenderFrame();
	}

}


//-----------------------------------------------------------------------------
// Purpose:
//-----------------------------------------------------------------------------
void CMainApplication::UpdateHMDMatrixPose()
{
	if ( !m_pHMD )
		return;

	vr::VRCompositor()->WaitGetPoses(m_rTrackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0 );

	m_iValidPoseCount = 0;
	m_strPoseClasses = "";
	for ( int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice )
	{
		if ( m_rTrackedDevicePose[nDevice].bPoseIsValid )
		{
			m_iValidPoseCount++;
			m_rmat4DevicePose[nDevice] = ConvertSteamVRMatrixToMatrix4( m_rTrackedDevicePose[nDevice].mDeviceToAbsoluteTracking );
			if (m_rDevClassChar[nDevice]==0)
			{
				switch (m_pHMD->GetTrackedDeviceClass(nDevice))
				{
				case vr::TrackedDeviceClass_Controller:        m_rDevClassChar[nDevice] = 'C'; break;
				case vr::TrackedDeviceClass_HMD:               m_rDevClassChar[nDevice] = 'H'; break;
				case vr::TrackedDeviceClass_Invalid:           m_rDevClassChar[nDevice] = 'I'; break;
				case vr::TrackedDeviceClass_GenericTracker:    m_rDevClassChar[nDevice] = 'G'; break;
				case vr::TrackedDeviceClass_TrackingReference: m_rDevClassChar[nDevice] = 'T'; break;
				default:                                       m_rDevClassChar[nDevice] = '?'; break;
				}
			}
			m_strPoseClasses += m_rDevClassChar[nDevice];
		}
	}

	if ( m_rTrackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid )
	{
		m_mat4HMDPose = m_rmat4DevicePose[vr::k_unTrackedDeviceIndex_Hmd];
		m_mat4HMDPose.invert();
	}
}

//-----------------------------------------------------------------------------
// Purpose: Converts a SteamVR matrix to our local matrix class
//-----------------------------------------------------------------------------
Matrix4 CMainApplication::ConvertSteamVRMatrixToMatrix4( const vr::HmdMatrix34_t &matPose )
{
	Matrix4 matrixObj(
		matPose.m[0][0], matPose.m[1][0], matPose.m[2][0], 0.0,
		matPose.m[0][1], matPose.m[1][1], matPose.m[2][1], 0.0,
		matPose.m[0][2], matPose.m[1][2], matPose.m[2][2], 0.0,
		matPose.m[0][3], matPose.m[1][3], matPose.m[2][3], 1.0f
		);
	return matrixObj;
}


//-----------------------------------------------------------------------------
// Purpose: Create/destroy GL Render Models
//-----------------------------------------------------------------------------
CGLRenderModel::CGLRenderModel( const std::string & sRenderModelName )
	: m_sModelName( sRenderModelName )
{
	m_glIndexBuffer = 0;
	m_glVertArray = 0;
	m_glVertBuffer = 0;
	m_glTexture = 0;
}


CGLRenderModel::~CGLRenderModel()
{
	Cleanup();
}


//-----------------------------------------------------------------------------
// Purpose: Allocates and populates the GL resources for a render model
//-----------------------------------------------------------------------------
bool CGLRenderModel::BInit( const vr::RenderModel_t & vrModel, const vr::RenderModel_TextureMap_t & vrDiffuseTexture )
{
	// create and bind a VAO to hold state for this model
	glGenVertexArrays( 1, &m_glVertArray );
	glBindVertexArray( m_glVertArray );

	// Populate a vertex buffer
	glGenBuffers( 1, &m_glVertBuffer );
	glBindBuffer( GL_ARRAY_BUFFER, m_glVertBuffer );
	glBufferData( GL_ARRAY_BUFFER, sizeof( vr::RenderModel_Vertex_t ) * vrModel.unVertexCount, vrModel.rVertexData, GL_STATIC_DRAW );

	// Identify the components in the vertex buffer
	glEnableVertexAttribArray( 0 );
	glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, sizeof( vr::RenderModel_Vertex_t ), (void *)offsetof( vr::RenderModel_Vertex_t, vPosition ) );
	glEnableVertexAttribArray( 1 );
	glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, sizeof( vr::RenderModel_Vertex_t ), (void *)offsetof( vr::RenderModel_Vertex_t, vNormal ) );
	glEnableVertexAttribArray( 2 );
	glVertexAttribPointer( 2, 2, GL_FLOAT, GL_FALSE, sizeof( vr::RenderModel_Vertex_t ), (void *)offsetof( vr::RenderModel_Vertex_t, rfTextureCoord ) );

	// Create and populate the index buffer
	glGenBuffers( 1, &m_glIndexBuffer );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_glIndexBuffer );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( uint16_t ) * vrModel.unTriangleCount * 3, vrModel.rIndexData, GL_STATIC_DRAW );

	glBindVertexArray( 0 );

	// create and populate the texture
	glGenTextures(1, &m_glTexture );
	glBindTexture( GL_TEXTURE_2D, m_glTexture );

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, vrDiffuseTexture.unWidth, vrDiffuseTexture.unHeight,
		0, GL_RGBA, GL_UNSIGNED_BYTE, vrDiffuseTexture.rubTextureMapData );

	// If this renders black ask McJohn what's wrong.
	glGenerateMipmap(GL_TEXTURE_2D);

	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

	GLfloat fLargest;
	glGetFloatv( GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, fLargest );

	glBindTexture( GL_TEXTURE_2D, 0 );

	m_unVertexCount = vrModel.unTriangleCount * 3;

	return true;
}


//-----------------------------------------------------------------------------
// Purpose: Frees the GL resources for a render model
//-----------------------------------------------------------------------------
void CGLRenderModel::Cleanup()
{
	if( m_glVertBuffer )
	{
		glDeleteBuffers(1, &m_glIndexBuffer);
		glDeleteVertexArrays( 1, &m_glVertArray );
		glDeleteBuffers(1, &m_glVertBuffer);
		m_glIndexBuffer = 0;
		m_glVertArray = 0;
		m_glVertBuffer = 0;
	}
}

//-----------------------------------------------------------------------------
// Purpose:
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
	CMainApplication *pMainApplication = new CMainApplication( argc, argv );

	if (!pMainApplication->BInit())
	{
		pMainApplication->Shutdown();
		return 1;
	}
	pMainApplication->RunMainLoop();
	pMainApplication->Shutdown();
	return 0;
}
