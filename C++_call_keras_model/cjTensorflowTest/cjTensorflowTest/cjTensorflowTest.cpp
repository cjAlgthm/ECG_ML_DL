//cjTensorflowTest.cpp : 定义控制台应用程序的入口点。
#include "stdafx.h"
#include <fstream>
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/public/session.h"
using namespace std;
using namespace tensorflow;

int main()
{
	printf("TF version is: %s\n", TF_Version());
	//return 0;

	//输入的数据和模型
	const string dataPath = "../Adata19_Noise.txt";
	ifstream dataFile(dataPath);
	const string modelPath = "../AfPredict.pb";

	//Read TXT data to array
	const int dataLen = 18176;// 128 * 60
	float *pdata = new float[dataLen];
	for (int i = 0; i < dataLen; i++)
	{
		dataFile >> pdata[i];
	}
	dataFile.close();

	// Fill input tensor with input data
	Tensor inputData(DT_FLOAT, TensorShape({1,dataLen,1}));
	auto inputData_mapped = inputData.tensor<float,3>();
	for (int i = 0; i < dataLen; i++)
	{
		inputData_mapped(0,i,0) = pdata[i];
	}
	if (pdata != NULL)
	{
		delete pdata;
	}

	//创建新会话session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);

   //从pb文件中读取模型
	GraphDef graphdef;
	Status status_load = ReadBinaryProto(Env::Default(), modelPath, &graphdef);
	if (!status_load.ok()) {
		cout << "ERROR: Loading model failed..." << modelPath << std::endl;
		cout << status_load.ToString() << "\n";
		return -1;
	}

	//将模型导入会话Session中;
	Status status_create = session->Create(graphdef); 
	if (!status_create.ok()) {
		cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
		return -1;
	}
	cout << "Successfully created session and load graph." << endl;

	//预测
	string InputName = "inputs";
	string OutputName = "activation_34/truediv";
	vector<pair<string,Tensor> > inputs;
	inputs.push_back(pair<string,Tensor>(InputName, inputData));
	vector<Tensor> outputs;

	Status status_run = session->Run(inputs, { OutputName }, {}, &outputs);
	if (!status_run.ok()) 
	{
		cout << "ERROR: RUN failed..." << std::endl;
		cout << status_run.ToString() << "\n";
		return -1;
	}

	//把输出值给提取出来
	cout << "Output tensor size:" << outputs.size() << std::endl;
	for (std::size_t i = 0; i < outputs.size(); i++) 
	{
		cout << outputs[i].DebugString() << endl;
	}

	Tensor t = outputs[0];                   // Fetch the first tensor
	int dataSeg = t.shape().dim_size(1);  
	int target_class_num = t.shape().dim_size(2);  // Get the target_class_num from 1st dimension
	auto tmap = t.tensor<float, 3>();        // Tensor Shape: [batch_size, target_class_num]

	// Argmax: Get Final Prediction Label and Probability
	//房颤的结果标识：0-A 1-N 2-O 3-Noise(符号是~)
	int i, j;
	for (i= 0; i< dataSeg; i++)
	{
		int output_class_id = -1;
		double output_prob = 0.0;
		cout << "DataSeg" << i << std::endl;

		for(j=0;j<target_class_num;j++)
		{
			cout << "Class " << j << " prob:" << tmap(0, i, j) << "," << std::endl;
			if (tmap(0, i, j) >= output_prob)
			{
				output_class_id = j;
				output_prob = tmap(0, i, j);
			}
		}
		// 输出结果
		cout << "Final class id: " << output_class_id << std::endl;
		cout << "Final class prob: " << output_prob << std::endl;
	}

	//关闭Session
	session->Close();
	getchar();
}

