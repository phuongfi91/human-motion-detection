#ifndef FTRACKER
#define FTRACKER

#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

// Số lượng các đặc trưng của mỗi trajectory.
#define MAX_FEATURES 4

using namespace cv;
using namespace std;

class FeatureTracker
{
private:
	Mat gray; // Frame hiện tại.
	Mat gray_prev; // Frame trước đó.
	Mat tempMat; // Biến tạm thời để lưu các đặc trưng của 1 trajectory.
	vector<Point2f> points[2]; // Các đặc trưng được track. Points[0] sẽ làm input, Points[1] làm output cho hàm track.
	vector<Point2f> features; // Danh sách các điểm đặc trưng detect được.
	vector<vector<Point2f>> trajectories; // Mảng chứa các Trajectories đang track (có chứa cả quỹ đạo)
	vector<vector<Point2f>> selectedTrajectories; // Mảng chứa các Trajectories được lựa chọn (có chứa cả quỹ đạo)
	int L; // Tuổi thọ tối đa của một điểm được track.
	int lowerbound; // Số lượng số point tối thiểu để track trong mỗi frame.
	int max_count; // Số lượng các điểm đặc trưng tối đa mà hàm goodFeaturesToTrack sẽ detect.
	float movement_sensibility; // Chỉ số cảm nhận chuyển động. Dùng để check xem một điểm có di chuyển sau mỗi frame hay không.
	double qlevel; // Chỉ số chất lượng trong hàm goodFeaturesToTrack.
	double minDist; // Khoảng cách tối thiểu giữa 2 điểm đặc trưng.
	vector<uchar> status; // Trạng thái của điểm được track sau mỗi frame. 1 là track thành công. 0 là điểm đã thất lạc, track thất bại.
	vector<float> err; // Mảng chứa các vector error.

public:
	vector<Mat> FeaturesSet; // Mảng chứa các đặc trưng của trajectory.

	// Constructor. Các thông số này đã được optimize để lấy được số lượng
	// đặc trưng vừa đủ.
	FeatureTracker() : L(10), lowerbound(5), movement_sensibility(1.5), max_count(100), qlevel(0.01), minDist(10.), tempMat(1, MAX_FEATURES, CV_32FC1) {}
	
	// Hàm xử lý chính để trích xuất đặc trưng của frame.
	void Process(Mat &frame,  Mat &output)
	{
		// Convert sang ảnh đen trắng.
		cvtColor(frame, gray, CV_BGR2GRAY); 
		frame.copyTo(output);
		if (!gray_prev.empty())
		{
			// Track điểm trong Points[0] và output ra Points[1], sử dụng
			// phương pháp của Lucas-Kanade.
			calcOpticalFlowPyrLK(
				gray_prev, gray, // Frame trước đó và hiện tại
				points[0], // Các điểm đang track ở frame trước đó
				points[1], // Output vị trí của các điểm được track ở frame hiện tại
				status, // Trạng thái track. 0 hoặc 1.
				err, // Mảng chứa các vector error.
				Size(15, 15)); // Vùng cửa sổ track trong phương pháp của Lucas-Kanade.

			// Kiểm tra các điểm đang được track xem điểm nào có thể được giữ lại
			// để track tiếp trong frame sau.
			int k = 0;
			int tsize = trajectories.size();
			for(int i = 0; i < tsize; i++)
			{
				// Nếu Trajectory đã đạt được tuổi thọ L thì tiến hành thêm vào mảng
				// NormalizedTrajectories.
				if (trajectories[i].size() == L)
				{
					float x, y;
					float sumDist = 0;
					for (int j = 1; j < L; j++)
					{
						x = trajectories[i][j].x - trajectories[i][j - 1].x;
						y = trajectories[i][j].y - trajectories[i][j - 1].y;
						sumDist += sqrt(x * x + y * y);
					}
					// Đặc trưng thứ nhất là vector chỉ hướng của trajectory
					// (Đã được normalized). Gồm 2 giá trị x và y.
					tempMat.at<float>(0,0) = (trajectories[i][L - 1].x - trajectories[i][0].x) / sumDist;
					tempMat.at<float>(0,1) = (trajectories[i][L - 1].y - trajectories[i][0].y) / sumDist;

					// Đặc trưng thứ 2 là độ dời trung bình của các trajectory.
					tempMat.at<float>(0,2) = sumDist / trajectories[i].size();

					// Đặc trưng thứ 3 là khoảng cách trung bình từ trajectory này
					// đến các trajectory khác có cùng L.
					tempMat.at<float>(0,3) = AverageDistanceToOtherTrajectories(i);

					// Nhét hết các đặc trưng này vào túi.
					FeaturesSet.push_back(tempMat.clone());

					// Nhét trajectory này vào một túi để lưu tạm thời.
					// Dùng cho việc tính toán khoảng cách trung bình giữa
					// các trajectory.
					selectedTrajectories.push_back(trajectories[i]);
				}

				// Kiểm tra xem Trajectory này có thỏa điều kiện giữ lại hay
				// không. Nếu có thì giữ lại và tiếp tục track.
				if (KeepThisPoint(i))
				{
					points[0][k] = points[0][i];
					points[1][k] = points[1][i];
					trajectories[k] = trajectories[i];
					trajectories[k].push_back(points[1][i]);
					k++;
				}
			}

			// Resize lại Trajectory để chứa vừa đủ các phần tử vừa giữ lại.
			points[0].resize(k);
			points[1].resize(k);
			trajectories.resize(k);

			// Vẽ các trajectories ra output.
			DrawTrajectories(frame, output);

			// Swap để biến cái hiện tại thành quá khứ.
			swap(points[1], points[0]);
			swap(gray_prev, gray);
		}
		// Frame đầu tiên sẽ có gray_prev là empty, ta thực hiện copy từ frame
		// hiện tại sang.
		else
		{
			gray.copyTo(gray_prev);
		}

		// Nếu cần thêm các điểm mới để track.
		if(NeedMorePoint())
		{
			// Detect các điểm đặc trưng trong frame sử dụng hàm goodFeaturesToTrack
			// đề xuất bởi Shi-Tomasi.
			goodFeaturesToTrack(
				gray, // Frame hiện tại cần detect
				features, // Danh sách chứa các điểm đặc trưng để output vào
				max_count, // Số lượng tối đa các đặc trưng
				qlevel, // Thông số chất lượng
				minDist); // Khoảng cách tối thiểu giữa 2 điểm đặc trưng

			// Add tất cả các điểm đặc trưng này vào cuối danh sách points[0]
			points[0].insert(points[0].end(), features.begin(), features.end());

			// Thêm các mảng vector con vào để chứa quỹ đạo của các Trajectories.
			int fsize = features.size();
			for (int i = 0; i < fsize; i++)
			{
				trajectories.push_back(vector<Point2f>());
				// Cập nhật quỹ đạo của các Trajectories.
				trajectories[trajectories.size() - 1].push_back(features[i]);
			}
		}
	}

	// Nếu cần thêm các điểm mới để track.
	bool NeedMorePoint()
	{
		// Vì số lượng đang track hiện tại quá ít.
		return points[0].size() <= lowerbound;
	}

	// Kiểm tra và giữ lại các điểm đang track mà thỏa mãn các điều kiện dưới đây.
	bool KeepThisPoint(int i)
	{
		return
			// Trạng thái có track được ở frame hiện tại hay không.
			status[i]
			&&
			// Điểm này có di chuyển so với frame trước hay không ?
			(abs(points[0][i].x-points[1][i].x)+
			(abs(points[0][i].y-points[1][i].y))
			> movement_sensibility)
			&&
			// Tuổi thọ của nó có bé hơn L hay không ?
			trajectories[i].size() < L;
	}

	// Vẽ quỹ đạo chuyển động của các điểm đang track.
	void DrawTrajectories( Mat &frame,  Mat &output)
	{
		int tisize;
		int tsize = trajectories.size();
		for(int i = 0; i < tsize; i++)
		{
			tisize = trajectories[i].size();
			if (tisize > 1)
			{
				for (int j = 1; j < tisize; j++)
				{
					line(output, trajectories[i][j - 1], trajectories[i][j], Scalar(0,255,0), 1, CV_AA);
				}
				circle(output, trajectories[i][tisize - 1], 2, Scalar(0,0,255),-1, CV_AA);
			}
		}
	}

	// Tính khoảng cách trung bình giữa các điểm trong trajectory hiện tại
	// so với các trajectory đã tìm được.
	float AverageDistanceToOtherTrajectories(int k)
	{
		float x, y;
		float delta = 0;
		int stsize = selectedTrajectories.size();
		for (int i = 0; i < stsize; i++)
		{
			for (int j = 0; j < L; j++)
			{
				x = selectedTrajectories[i][j].x - trajectories[k][j].x;
				y = selectedTrajectories[i][j].y - trajectories[k][j].y;
				delta += sqrt(x * x + y * y);
			}
		}
		if (stsize > 0)
		{
			return delta /= stsize;
		}
		return -1.0f;
	}
};

#endif
