#include <iostream>
#include <fstream>

#include "Eigen.h"

#include "VirtualSensor.h"

/* 
	Group Members
	Katam Harinandan Teja - 03710066
	Meng Liu - 03709787
*/
struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

// Checks if all the co-ordinates are valid (i.e. not inf)
bool isVectorValid(Vector4f vec) {
	return (vec.x() != MINF && vec.y() != MINF && vec.z() != MINF);
}

// Returns the distance between two vectors
float distance(Vector4f a, Vector4f b) {
	return sqrtf(pow(a.x() - b.x(), 2) + pow(a.y() - b.y(), 2) + pow(a.z() - b.z(), 2));
}

// Checks if a set of points can form a valid face. 
// Check if the vertices are valid and if all the edges have length less than the threshold
bool canBeAFace(Vector4f a, Vector4f b, Vector4f c, float edgeThreshold) {

	if (isVectorValid(a) && isVectorValid(b) && isVectorValid(c)) {

		float ab = distance(a, b);
		float bc = distance(b, c);
		float ca = distance(c, a);

		if (ab > edgeThreshold || bc > edgeThreshold || ca > edgeThreshold) {
			return false;
		}
		else {
			return true;
		}
	}
	else {
		return false;
	}

}

bool isValidIndex(int idx, int maxLength) {
	return idx > 0 && idx < maxLength;
}

bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename)
{
	float edgeThreshold = 0.01f; // 1cm

	// TODO 2: use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
	// - have a look at the "off_sample.off" file to see how to store the vertices and triangles
	// - for debugging we recommend to first only write out the vertices (set the number of faces to zero)
	// - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
	// - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
	// - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
	// - only write triangles with valid vertices and an edge length smaller then edgeThreshold

	// TODO: Get number of vertices
	unsigned int nVertices = width * height;

	// TODO: Get number of faces
	unsigned nFaces = width * height /2;

	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << std::endl;
	outFile << nVertices << " " << nFaces << " 0" << std::endl;

	// TODO: save vertices
	// Stores the vertices (X, Y, Z, R, G, B, A) in the file
	for (int idx = 0; idx < nVertices; idx++) {
		float positionX = vertices[idx].position.x();
		float positionY = vertices[idx].position.y();
		float positionZ = vertices[idx].position.z();

		if (positionX == MINF || positionY == MINF || positionZ == MINF) {
			outFile << "0 0 0 0 0 0" << std::endl;
		}
		else {
			int r = vertices[idx].color.x();
			int g = vertices[idx].color.y();
			int b = vertices[idx].color.z();
			int a = 255;

			outFile << positionX << " " << positionY << " " << positionZ << " " << r << " " << g << " " << b << " " << a << std::endl;
		}
	}

	// TODO: save faces
	// Stores the faces in the file. The faces are formed in the following way
	// For a vertex v let t be the vertex that is on top of the vertex v in the pixel coordinate system, l be the left, r be the right, b be the bottom. 
	// Then vgr and vtl are the two faces that are generated for this vertex v.
	// These faces are written to the file only if the faces are valid
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			int currentIdx = i * width + j;
			Vector4f current = vertices[currentIdx].position;

			if (current.x() != MINF && current.y() != MINF && current.z() != MINF) {
				int rightIdx = i * width + j + 1;
				int bottomIdx = (i + 1)*width + j;
				int leftIdx = i * width + j - 1;
				int topIdx = (i - 1)*width + j;
				
				if (isValidIndex(rightIdx, nVertices) && isValidIndex(bottomIdx, nVertices)) {
					Vector4f right = vertices[rightIdx].position;
					Vector4f bottom = vertices[bottomIdx].position;

					if (canBeAFace(current, right, bottom, edgeThreshold)) {
						outFile << "3 " << currentIdx << " " << bottomIdx << " " << rightIdx << std::endl;
					}
				}

				if (isValidIndex(leftIdx, nVertices) && isValidIndex(topIdx, nVertices)) {
					Vector4f left = vertices[leftIdx].position;
					Vector4f top = vertices[topIdx].position;

					if (canBeAFace(current, left, top, edgeThreshold)) {
						outFile << "3 " << currentIdx << " " << topIdx << " " << leftIdx << std::endl;
					}
				}
			}
		}
	}

	// close file
	outFile.close();

	return true;
}

int main()
{
	std::string filenameIn = "./data/rgbd_dataset_freiburg1_xyz/";
	std::string filenameBaseOut = "mesh_";

	// load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.Init(filenameIn))
	{
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// convert video to meshes
	while (sensor.ProcessNextFrame())
	{
		// get ptr to the current depth frame
		// depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
		float* depthMap = sensor.GetDepth();
		// get ptr to the current color frame
		// color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
		BYTE* colorMap = sensor.GetColorRGBX();

		// get depth intrinsics
		Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
		float fovX = depthIntrinsics(0, 0);
		float fovY = depthIntrinsics(1, 1);
		float cX = depthIntrinsics(0, 2);
		float cY = depthIntrinsics(1, 2);

		// compute inverse depth extrinsics
		Matrix4f depthExtrinsicsInv = sensor.GetDepthExtrinsics().inverse();

		Matrix4f trajectory = sensor.GetTrajectory();
		Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();

		// TODO 1: back-projection
		// write result to the vertices array below, keep pixel ordering!
		// if the depth value at idx is invalid (MINF) write the following values to the vertices array
		// vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
		// vertices[idx].color = Vector4uc(0,0,0,0);
		// otherwise apply back-projection and transform the vertex to world space, use the corresponding color from the colormap
		Vertex* vertices = new Vertex[sensor.GetDepthImageWidth() * sensor.GetDepthImageHeight()];

		// Get the depth image width and height
		int imageWidth = sensor.GetDepthImageWidth();
		int imageHeight = sensor.GetDepthImageHeight();

		// Loop through every pixel. i is the y pixel co-ordinate and j is the x pixel co-ordinate
		for (int i = 0; i < imageHeight; i++) {
			for (int j = 0; j < imageWidth; j++) {

				// Get the index of the pixel in the array
				int idx = i * imageWidth + j;
				float depth = depthMap[idx];

				// If depth if invalid
				if (depth == MINF) {
					vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
					vertices[idx].color = Vector4uc(0, 0, 0, 0);
				}
				else {
					float x = ((float)(j) - cX) / fovX;
					float y = ((float)(i) - cY) / fovY;

					// Get the co-ordinates in the camera co-ordinate system
					Vector4f cameraSytemCoordinates = Vector4f(depth*x, depth*y, depth, 1);
					// Apply the camera transformation
					Vector4f worldSpaceCoordinates = trajectoryInv * cameraSytemCoordinates;

					// Get the color(RGBA) from the colormap
					unsigned char r = colorMap[idx*4];
					unsigned char g = colorMap[idx*4 + 1];
					unsigned char b = colorMap[idx*4 + 2];
					unsigned char a = colorMap[idx*4 + 3];

					// Assign the position and the color to the vertex
					vertices[idx].position = worldSpaceCoordinates;
					vertices[idx].color = Vector4uc(r, g, b, a);
				}
			}
		}

		// write mesh file
		std::stringstream ss;
		ss << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
		if (!WriteMesh(vertices, sensor.GetDepthImageWidth(), sensor.GetDepthImageHeight(), ss.str()))
		{
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return -1;
		}

		// free mem
		delete[] vertices;
	}

	return 0;
}
