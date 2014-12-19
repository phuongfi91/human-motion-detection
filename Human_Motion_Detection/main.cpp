//////////////////////////////////////////////////////////////////////////
/* Human Motion Detection
 * Team CNTN04:
 * + Nguyễn Đăng Châu - 09520019
 * + Nguyễn Thành Luân - 09520163
 * + Bùi Tấn Phát - 09520601
 * + Nguyễn Dũng Phương - 09520607
 */
/* Built using OpenCV 2.3.1 */
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include "featuretracker.h"

using namespace std;

ifstream fin;
ofstream fout;
string motiontypes[]={ "", "Boxing", "Handclapping", "Handwaving", "Jogging", "Running", "Walking" };

// Hàm phụ trợ: Tiến hành trích xuất các đặc trưng của frame hiện tại, xuất ra output và
// lưu các thông tin đặc trưng trong tracker.
bool ExtractFeatures(Mat &frame, Mat &output, VideoCapture &capture, FeatureTracker &tracker) 
{
	capture >> frame;
	if (frame.empty())
	{
		return false;
	}
	tracker.Process(frame, output);
	return true;
}

// Hàm phụ trợ: Chuyển chuỗi về lowercase.
string LowerCase(string str)
{
	for (int i = 0; i < str.length(); i++) 
		str[i] = tolower(str[i]);
	return str;
}

// Hàm phụ trợ: Đọc kiểu hành động từ tên file. Dùng để đánh label cho bước
// máy học và dùng để xác định xem việc nhận dạng là đúng hay sai.
int GetLabel(string &strinput) 
{
	size_t found;
	for (int i = 1; i <= 6; i++)
	{
		found = strinput.find(LowerCase(motiontypes[i]));
		if (found != string::npos)
		{
			return i;
		}
	}
	return 0;
}

// Hàm phụ trợ: Thông báo lỗi mở file.
void ErrorOpening(string name)
{
	cout << "Error: Cannot open " + name + " !\n";
}

// Hàm chính: Xuất các video trong input ra màn hình và vẽ quỹ đạo chuyển động.
void OutputVideo(string input)
{
	string strinput;

	fin.open(input, ifstream::in);
	if (fin.fail()) ErrorOpening(input);
	else
	{
		cout << "Press ESC key to terminate video output.\n";
		// Đọc từng file trong từng dòng của input.
		while(getline(fin, strinput))
		{
			VideoCapture capture(strinput);
			if (!capture.isOpened())
			{
				ErrorOpening(strinput);
				continue;
			}
			
			FeatureTracker tracker;

			namedWindow(strinput);

			// Lấy framerate của file video.
			double framerate = capture.get(CV_CAP_PROP_FPS);

			while(1)
			{
				// Trích xuất các quỹ đạo.
				Mat frame, output;
				if (!ExtractFeatures(frame, output, capture, tracker)) break;

				// Và xuất video ra màn hình.
				imshow(strinput, output);

				// ESC để hủy.
				char key = (char)waitKey(framerate);
				switch (key)
				{
				case 27: // ESC key
					destroyWindow(strinput);
					cout << "Video output has been terminated by user.\n";
					fin.close();
					return;
				default:
					break;
				}
			}
			destroyWindow(strinput);
		}
		cout << "Video output completed.\n";
	}
	fin.close();
}

// Hàm chính: Tiến hành nhận dạng các video trong input bằng cách sử dụng dữ
// liệu máy học. Đồng thời output kết quả ra file.
void Recognize(string input, string output, string trainingData)
{
	int failedSamples = 0; // Số lượng mẫu đã nhận sai.
	int processedSamples = 0; // Số lượng mẫu đã đọc.
	string strinput;

	// Check file dữ liệu máy học.
	fin.open(trainingData);
	if (fin.fail())
	{
		ErrorOpening(trainingData);
		return;
	}
	else fin.close();

	// Mở file input.
	fin.open(input, ifstream::in);
	if (fin.fail()) ErrorOpening(input);
	else
	{
		cout << "Recognizing...\n";
		fout.open(output, ofstream::out);

		// Load dữ liệu máy học.
		CvSVM SVM;
		SVM.load(trainingData.c_str());

		// Đọc từng file trong từng dòng của input.
		while(getline(fin, strinput))
		{
			VideoCapture capture(strinput);
			if (!capture.isOpened())
			{
				ErrorOpening(strinput);
				fout << "Error: Cannot open " + strinput + " !\n";
				continue;
			}

			processedSamples++;

			FeatureTracker tracker;
			vector<float> responses; // Response từ SVM. (Trả về label từ 1->6)

			while(1)
			{
				Mat frame, output;
				if (!ExtractFeatures(frame, output, capture, tracker)) break;
			}

			int fsize = tracker.FeaturesSet.size();
			int responsesCounter[7] = { 0,0,0,0,0,0,0 }; // Mảng đếm responses.
			for (int i = 0; i < fsize; i++)
				responsesCounter[(int)(SVM.predict(tracker.FeaturesSet[i]))]++;

			// Tìm response xuất hiện nhiều nhất và kết luận hành động.
			int max = 1;
			for (int i = 2; i < 7; i++)
				if (responsesCounter[i] > responsesCounter[max]) max = i;

			// So sánh kết luận này với tên file để biết được nhận dạng là
			// đúng hay sai.
			if (max != GetLabel(strinput)) failedSamples++;

			// Xuất ra màn hình và ra file.
			fout << strinput << " -> " << motiontypes[max] << endl;
			cout << strinput << " -> " << motiontypes[max] << endl;
		}

		cout<<"Action recognition completed. Results has been saved to " + output + " .\n";
		cout<<"Overall accuracy: "<< setiosflags(ios::fixed) << setprecision(2) 
			<< (1.0f - ((float)failedSamples / processedSamples)) * 100 <<" %\n";
	}
	fin.close();
	fout.close();
}

// Hàm chính: Tiến hành cho máy học các video từ input (tên video cần có tên
// hành động tương ứng để máy đánh label cho phù hợp.
void Train(string input, string trainingData)
{
	string strinput;

	fin.open(input, ifstream::in);
	if (fin.fail()) ErrorOpening(input);
	else
	{
		vector<float> labels;
		vector<Mat> features;

		// Đọc từng file trong từng dòng của input.
		while(getline(fin, strinput))
		{
			VideoCapture capture(strinput);
			if (!capture.isOpened())
			{
				ErrorOpening(strinput);
				continue;
			}

			// Lấy label (từ 1->6) từ tên file.
			int label = GetLabel(strinput);

			// Không thể rút ra label từ tên file.
			if (label == 0)
			{
				cout << "Error: Unrecognized Action Label of " + strinput + " !\n";
				continue;
			}

			cout << "Extracting features from " + strinput + "...\n";

			FeatureTracker tracker;

			while(1)
			{
				Mat frame, output;
				if (!ExtractFeatures(frame, output, capture, tracker)) break;
			}
			
			// Rút các đặc trưng của video này vào túi chứa tổng hợp đặc trưng.
			int fsize = tracker.FeaturesSet.size();
			for (int i = 0; i < fsize; i++)
			{
				labels.push_back(label);
				features.push_back(tracker.FeaturesSet[i]);
			}
		}

		cout << "Features extraction completed. Training...\n";

		// Chuyển hết các đặc trưng và label sang dạng Mat để dùng cho máy học SVM.
		Mat labelsMat;
		labelsMat.create(labels.size(), 1, CV_32FC1);
		Mat trainingDataMat;
		trainingDataMat.create(features.size(), 4, CV_32FC1);
		int fsize = features.size();
		for (int i = 0; i < fsize; i++)
		{
			labelsMat.at<float>(i,0) = labels[i];
			for (int j = 0; j < MAX_FEATURES; j++)
				trainingDataMat.at<float>(i,j) = features[i].at<float>(0,j);
		}

		// Xây dựng params chứa các thông số điều khiển SVM.
		CvSVMParams params;
		params.svm_type    = CvSVM::C_SVC;
		params.C 		   = 0.1;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);

		// Tiến hành training và save lại dữ liệu ra file.
		CvSVM SVM;
		SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
		SVM.save(trainingData.c_str());
		cout << "Training completed. Training data has been saved to " + trainingData + "\n";
	}
	fin.close();
}

// Hàm phụ trợ: Hàm input nhằm tránh người dùng nhập vào nội dung khác số.
void CorrectInput(int &i)
{
	while (!(cin>>i))
	{
		cin.clear(); // Reset input.
		while (cin.get() != '\n') continue; // Xóa hết các kí tự input sai.
		break;
	}
}

// Hàm chính: Vẽ màn hình chọn chức năng cho người sử dụng
void main()
{
	while(1)
	{
		// Option.
		int choice;

		cout<<"*******************************************************************************\n";
		cout<<"*                           HUMAN MOTION DETECTION                            *\n";
		cout<<"*******************************************************************************\n\n";

		// Các options:
		cout<<"\nPlease choose an option:\n";
		cout<<"1.  Output video files (with trajectories) listed in input.txt\n";
		cout<<"2.  Recognize video files listed in input.txt using training_data.xml\n";
		cout<<"3.  Train video files listed in input.txt and output to training_data.xml\n";
		cout<<"4.  Exit program.\n\n";
		cout<<"-------------------------------------------------------------------------------\n";
		cout<<"Please enter your option:";

		// Xử lý input.
		CorrectInput(choice);

		switch(choice)
		{
		case 1:OutputVideo("input.txt"); break;
		case 2:Recognize("input.txt", "output.txt", "training_data.xml"); break;
		case 3:Train("input.txt", "training_data.xml"); break;
		case 4:exit(1); break;
		default:cout << "Please choose the correct option...\n"; break;
		}

		system("pause");
		system("cls");
	}
}