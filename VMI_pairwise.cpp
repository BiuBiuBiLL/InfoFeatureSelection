#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <map>
#include <set>
using namespace std;

double const reload_threshold = 1e-2;
int const Nsamples =  73;
int const Nfeatures = 325;
int const Nclasses = 7;
int const selected_features = 100;

string const file_name = "data/penglungEW";

short int data[Nsamples][Nfeatures];
short int label[Nsamples];
typedef struct jpState
{
  double *jointProbabilityVector;
  int numJointStates;
  double *firstProbabilityVector;
  int numFirstStates;
  double *secondProbabilityVector;
  int numSecondStates;
  map<double, int> firstStateMap;
  map<double, int> secondStateMap;
} JointProbabilityState;
double nominator[Nfeatures];
double x_matrix[Nsamples][Nclasses];
double prob_ratio[Nfeatures][Nsamples][Nclasses];
double memory_store[Nfeatures][Nsamples][Nclasses];
double featureMergeArray[Nfeatures][Nsamples];
double data_inverse[Nfeatures][Nsamples];

inline float fast_log2 (float val)
{
   int * const    exp_ptr = reinterpret_cast <int *> (&val);
   int            x = *exp_ptr;
   const int      log_2 = ((x >> 23) & 255) - 128;
   x &= ~(255 << 23);
   x += 127 << 23;
   *exp_ptr = x;

   val = ((-1.0f/3) * val + 2) * val - 2.0f/3;   // (1)

   return (val + log_2);
} 

int normaliseArray(double *inputVector, int *outputVector, int vectorLength, map<double, int>& stateMap)
{
  int minVal = 0;
  int maxVal = 0;
  int currentValue;
  int i;
  stateMap.clear();
  
  if (vectorLength > 0)
  {
    minVal = (int) floor(inputVector[0]);
    maxVal = (int) floor(inputVector[0]);
  
    for (i = 0; i < vectorLength; i++)
    {
      currentValue = (int) floor(inputVector[i]);
      outputVector[i] = currentValue;
      
      if (currentValue < minVal)
      {
        minVal = currentValue;
      }
      else if (currentValue > maxVal)
      {
        maxVal = currentValue;
      }
    }/*for loop over vector*/
    
    for (i = 0; i < vectorLength; i++)
    {
      outputVector[i] = outputVector[i] - minVal;
      stateMap[inputVector[i]] = outputVector[i];
    }

    maxVal = (maxVal - minVal) + 1;
  }
  
  return maxVal;
}/*normaliseArray(double*,double*,int)*/
int mergeArrays(double *firstVector, double *secondVector, double *outputVector, int vectorLength)
{
  int *firstNormalisedVector;
  int *secondNormalisedVector;
  int firstNumStates;
  int secondNumStates;
  int i;
  int *stateMap;
  int stateCount;
  int curIndex;
  
  firstNormalisedVector = (int *) calloc(vectorLength,sizeof(int));
  secondNormalisedVector = (int *) calloc(vectorLength,sizeof(int));

  map<double, int> state_map;
  firstNumStates = normaliseArray(firstVector,firstNormalisedVector,vectorLength,state_map);
  secondNumStates = normaliseArray(secondVector,secondNormalisedVector,vectorLength,state_map);
  
  /*
  ** printVector(firstNormalisedVector,vectorLength);
  ** printVector(secondNormalisedVector,vectorLength);
  */
  stateMap = (int *) calloc(firstNumStates*secondNumStates,sizeof(int));
  stateCount = 1;
  for (i = 0; i < vectorLength; i++)
  {
    curIndex = firstNormalisedVector[i] + (secondNormalisedVector[i] * firstNumStates);
   /*
    if (stateMap[curIndex] == 0)
    {
      stateMap[curIndex] = stateCount;
      stateCount++;
    }
*/
    outputVector[i] = curIndex;
  }
    
  free(firstNormalisedVector);
  free(secondNormalisedVector);
  free(stateMap);
  
  firstNormalisedVector = NULL;
  secondNormalisedVector = NULL;
  stateMap = NULL;
  
  /*printVector(outputVector,vectorLength);*/
  return stateCount;
}/*mergeArrays(double *,double *,double *, int, bool)*/

JointProbabilityState calculateJointProbability(double *firstVector, double *secondVector, int vectorLength)
{
  int *firstNormalisedVector;
  int *secondNormalisedVector;
  int *firstStateCounts;
  int *secondStateCounts;
  int *jointStateCounts;
  double *firstStateProbs;
  double *secondStateProbs;
  double *jointStateProbs;
  int firstNumStates;
  int secondNumStates;
  int jointNumStates;
  int i;
  double length = vectorLength;
  map<double, int> firstStateMap;
  map<double, int> secondStateMap;

  JointProbabilityState state;
  firstNormalisedVector = (int *) calloc(vectorLength,sizeof(int));
  secondNormalisedVector = (int *) calloc(vectorLength,sizeof(int));
  
  firstNumStates = normaliseArray(firstVector,firstNormalisedVector,vectorLength, firstStateMap);
  secondNumStates = normaliseArray(secondVector,secondNormalisedVector,vectorLength, secondStateMap);
  jointNumStates = firstNumStates * secondNumStates;
  
  firstStateCounts = (int *) calloc(firstNumStates,sizeof(int));
  secondStateCounts = (int *) calloc(secondNumStates,sizeof(int));
  jointStateCounts = (int *) calloc(jointNumStates,sizeof(int));
  
  firstStateProbs = (double *) calloc(firstNumStates,sizeof(double));
  secondStateProbs = (double *) calloc(secondNumStates,sizeof(double));
  jointStateProbs = (double *) calloc(jointNumStates,sizeof(double));
    
  for (i = 0; i < vectorLength; i++)
  {
    firstStateCounts[firstNormalisedVector[i]] += 1;
    secondStateCounts[secondNormalisedVector[i]] += 1;
    jointStateCounts[secondNormalisedVector[i] * firstNumStates + firstNormalisedVector[i]] += 1;
  }
  
  for (i = 0; i < firstNumStates; i++)
  {
    firstStateProbs[i] = firstStateCounts[i] / length;
  }
  
  for (i = 0; i < secondNumStates; i++)
  {
    secondStateProbs[i] = secondStateCounts[i] / length;
  }
  
  for (i = 0; i < jointNumStates; i++)
  {
    jointStateProbs[i] = jointStateCounts[i] / length;
  }

  free(firstNormalisedVector);
  free(secondNormalisedVector);
  free(firstStateCounts);
  free(secondStateCounts);
  free(jointStateCounts);
    
  firstNormalisedVector = NULL;
  secondNormalisedVector = NULL;
  firstStateCounts = NULL;
  secondStateCounts = NULL;
  jointStateCounts = NULL;
  
  
  state.jointProbabilityVector = jointStateProbs;
  state.numJointStates = jointNumStates;
  state.firstProbabilityVector = firstStateProbs;
  state.numFirstStates = firstNumStates;
  state.secondProbabilityVector = secondStateProbs;
  state.numSecondStates = secondNumStates;
  state.firstStateMap = firstStateMap;
  state.secondStateMap = secondStateMap;
  return state;
}/*calculateJointProbability(double *,double *, int)*/


void readData(const string& train_file_name, const string& train_label_file_name) {
	string line;
	ifstream train_file(train_file_name), label_file(train_label_file_name);
	int i = 0, j = 0;
	while (getline(train_file, line)) {
		stringstream ss(line);
		string tem;
		j = 0;
		while (getline(ss, tem, '\t')) {
			data[i][j++] = stoi(tem);
		}
		++i;
	}
	i = 0;
	while (getline(label_file, line)) {
		label[i++] = stoi(line);	
	}
}

//log of conditional probability p(fi|se)=p(fi,se)/p(se)
double log_probability_con(JointProbabilityState& s, int &fi, int &se) 
{
	if (s.secondProbabilityVector[se] < 1e-10) 
		return 0.0;
	double value = s.jointProbabilityVector[se*s.numFirstStates+fi] / s.secondProbabilityVector[se];
	if (value < 1e-10) {
		return -1e10;
	} else {
		return log(value);
	}
}
inline double probability_con(JointProbabilityState& s, int &fi, int &se) 
{
	if (s.secondProbabilityVector[se] < 1e-10) 
		return 1.0;
	double value = s.jointProbabilityVector[se*s.numFirstStates+fi] / s.secondProbabilityVector[se];
	return value;
}

//logsumexp
double logsumexp(double nums[], int ct) {
  double max_exp = nums[0], sum = 0.0;
  size_t i;

  for (i = 1 ; i < ct ; i++)
    if (nums[i] > max_exp)
      max_exp = nums[i];

  for (i = 0; i < ct ; i++)
    sum += exp(nums[i] - max_exp);

  return log(sum) + max_exp;
}


int filtered[501] = {2, 12, 16, 34, 60, 70, 83, 95, 101, 112, 139, 155, 164, 189, 195, 205, 214, 225, 229, 246, 286, 288, 295, 300, 312, 328, 329, 338, 348, 356, 364, 365, 367, 378, 379, 393, 401, 409, 439, 443, 455, 467, 471, 487, 495, 497, 499, 509, 511, 518, 526, 532, 537, 554, 557, 568, 576, 579, 592, 593, 596, 600, 616, 629, 637, 641, 651, 657, 665, 668, 681, 682, 698, 719, 729, 756, 757, 764, 776, 796, 813, 818, 834, 840, 841, 846, 851, 855, 858, 865, 875, 879, 880, 884, 893, 902, 904, 916, 934, 948, 976, 981, 986, 1030, 1032, 1049, 1067, 1095, 1109, 1125, 1128, 1133, 1155, 1175, 1180, 1186, 1212, 1222, 1228, 1243, 1258, 1265, 1271, 1275, 1277, 1282, 1293, 1328, 1332, 1349, 1358, 1359, 1376, 1387, 1388, 1406, 1407, 1424, 1479, 1495, 1500, 1504, 1506, 1511, 1522, 1532, 1536, 1539, 1546, 1556, 1558, 1565, 1575, 1576, 1586, 1588, 1592, 1599, 1600, 1606, 1638, 1642, 1652, 1654, 1656, 1663, 1682, 1693, 1697, 1708, 1709, 1718, 1732, 1761, 1762, 1771, 1787, 1793, 1794, 1813, 1821, 1837, 1853, 1860, 1861, 1863, 1870, 1873, 1881, 1905, 1908, 1909, 1922, 1927, 1929, 1936, 1954, 1966, 1967, 1980, 1987, 1995, 2007, 2016, 2051, 2052, 2056, 2068, 2094, 2098, 2099, 2101, 2104, 2116, 2123, 2132, 2141, 2143, 2152, 2169, 2175, 2187, 2189, 2222, 2235, 2244, 2253, 2282, 2301, 2305, 2314, 2316, 2325, 2343, 2353, 2354, 2366, 2367, 2368, 2378, 2381, 2392, 2401, 2404, 2421, 2422, 2426, 2427, 2471, 2474, 2479, 2483, 2488, 2489, 2491, 2509, 2513, 2554, 2559, 2571, 2589, 2615, 2621, 2628, 2634, 2638, 2641, 2660, 2662, 2673, 2676, 2682, 2714, 2740, 2742, 2764, 2767, 2769, 2782, 2783, 2784, 2801, 2812, 2831, 2835, 2886, 2887, 2895, 2917, 2926, 2930, 2960, 2962, 2971, 2996, 3002, 3010, 3011, 3031, 3046, 3051, 3057, 3062, 3065, 3066, 3073, 3084, 3105, 3109, 3122, 3133, 3149, 3153, 3162, 3171, 3172, 3187, 3188, 3197, 3223, 3248, 3251, 3253, 3266, 3269, 3275, 3283, 3302, 3304, 3316, 3320, 3327, 3347, 3355, 3359, 3360, 3365, 3372, 3376, 3385, 3418, 3426, 3443, 3450, 3463, 3468, 3491, 3508, 3514, 3518, 3532, 3543, 3560, 3585, 3587, 3603, 3605, 3637, 3642, 3643, 3647, 3648, 3656, 3658, 3666, 3670, 3688, 3694, 3699, 3700, 3707, 3719, 3721, 3725, 3730, 3737, 3746, 3752, 3755, 3776, 3796, 3803, 3827, 3833, 3846, 3851, 3857, 3865, 3876, 3887, 3893, 3902, 3951, 3958, 3965, 3970, 3975, 4000, 4021, 4045, 4062, 4094, 4097, 4104, 4105, 4106, 4109, 4114, 4116, 4129, 4146, 4158, 4164, 4168, 4183, 4187, 4188, 4195, 4197, 4202, 4228, 4239, 4267, 4271, 4275, 4290, 4330, 4353, 4365, 4379, 4386, 4390, 4403, 4409, 4412, 4416, 4424, 4433, 4445, 4447, 4456, 4466, 4486, 4493, 4503, 4507, 4543, 4553, 4567, 4575, 4585, 4588, 4596, 4608, 4610, 4642, 4652, 4655, 4660, 4681, 4689, 4690, 4694, 4721, 4733, 4761, 4779, 4788, 4802, 4808, 4831, 4832, 4835, 4844, 4856, 4862, 4865, 4869, 4875, 4876, 4878, 4891, 4893, 4906, 4916, 4917, 4922, 4924, 4925, 4934, 4936, 4941, 4949, 4963, 4966, 4967, 4976, 4980, 4981, 4991, 4999};

JointProbabilityState featureDistribution[Nfeatures];

void feature_selection_discrete_greedy(int n_features) {
	set<int> selected_feature_set;
	for (int i = 0; i < 501; ++i) {
		selected_feature_set.insert(filtered[i]);
	}
	double *label_vector = (double*) calloc(Nsamples,sizeof(double));
	for (int i = 0; i < Nsamples; ++i) {
		label_vector[i] = label[i];
	}

	double *cur_feature = (double*) calloc(Nsamples, sizeof(double));
	for (int i = 0; i < Nfeatures; ++i) {
		for (int j = 0; j < Nsamples; ++j) {
			cur_feature[j] = data[j][i];
			data_inverse[i][j] = data[j][i];
		}
		//calculateJointProbability(cur_feature, label_vector, Nsamples);
		featureDistribution[i] = calculateJointProbability(cur_feature, label_vector, Nsamples);
		mergeArrays(cur_feature, label_vector, featureMergeArray[i], Nsamples);
	}
	//calculate nominator sum of p(x[j][i]|y=y[j]) for ith feature of all samples j=1..n
	for (int i = 0; i < Nfeatures; ++i) {
		for (int j = 0; j < Nsamples; ++j) {
			int firstValue = featureDistribution[i].firstStateMap[data[j][i]];
			int secondValue = featureDistribution[i].secondStateMap[label_vector[j]];
			nominator[i] += log_probability_con(featureDistribution[i], firstValue, secondValue);
		}
		nominator[i] /= Nsamples;
	}

	//each row of x_matrix stores the product of selected features ratio for each y, orginially, it is p(y)
	for (int i = 0; i < Nsamples; ++i) {
		for (int j = 0; j < featureDistribution[0].numSecondStates; ++j) {
			x_matrix[i][j] = featureDistribution[0].secondProbabilityVector[j];
		}
	}

	//ratio
	for (int i = 0; i < Nfeatures; ++i)
		for (int j = 0; j < Nsamples; ++j)
			for (int k = 0; k < featureDistribution[i].numSecondStates; ++k) {
				int firstValue = featureDistribution[i].firstStateMap[data[j][i]];
				prob_ratio[i][j][k] = probability_con(featureDistribution[i], firstValue ,k);
				memory_store[i][j][k] = 1.0;
			}
	//start feature selection
	int feature_array[Nfeatures];	
	set<int> feature_set;
	double MI_learned[Nfeatures];
	double last_MI_learned = 0.0;
	double last_nominator = 0.0;
	int cur_step = 0;
	int initial_step = 0;
	int top_feature = -1;
	int last_feature = -1;
	while (cur_step < n_features) {
		double max_MI_learned = -1e10;
		int selected_feature = -1;
		double max_nominator = 0.0;
		if (initial_step == 0) { //first step: select maximum feature
			for (int i = 0; i < Nfeatures; ++i) {
				if (feature_set.find(i) == feature_set.end()){ //&& selected_feature_set.find(i) != selected_feature_set.end()) {                                              
					double new_MI_learned = 0.0;
					for (int j = 0; j < Nsamples; ++j) {
						double tem = 0.0;
						for (int k = 0; k < featureDistribution[i].numSecondStates; ++k) {
							tem += x_matrix[j][k]*prob_ratio[i][j][k];
											
						}
						new_MI_learned -= log(tem);
					} 
					new_MI_learned /= Nsamples;
					new_MI_learned += last_nominator + nominator[i];
					if (new_MI_learned > max_MI_learned) {
						max_MI_learned = new_MI_learned;
						max_nominator = nominator[i];
						selected_feature = i;
					}
				}
			}
		} else {
			for (int i = 0; i < Nfeatures; ++i) {
				if (feature_set.find(i) == feature_set.end()){ //  && selected_feature_set.find(i) != selected_feature_set.end()) {
					last_feature = feature_array[cur_step-1];
					JointProbabilityState distribution = calculateJointProbability(data_inverse[i], featureMergeArray[last_feature], Nsamples);
					double new_MI_learned = 0.0;
					double nominator_sum = 0.0;
					for (int j = 0; j < Nsamples; ++j) {
						double tem = 0.0;
						for (int k = 0; k < featureDistribution[i].numSecondStates; ++k) {
							int firstValue = featureDistribution[i].firstStateMap[data[j][i]];
							int secondValue = featureDistribution[last_feature].firstStateMap[data[j][last_feature]]+k*featureDistribution[last_feature].numFirstStates;
							double conditional_prob = probability_con(distribution, firstValue, secondValue);
							memory_store[i][j][k] = pow(memory_store[i][j][k], (double)(initial_step-1.0)/(double)initial_step);
							memory_store[i][j][k] *= pow(conditional_prob, 1.0/(double)initial_step);
							tem += x_matrix[j][k]*memory_store[i][j][k];
							if (featureDistribution[i].secondStateMap[label_vector[j]] == k)
								nominator_sum += log(memory_store[i][j][k]);					
						}
						new_MI_learned -= log(tem);
					}
					new_MI_learned /= Nsamples;
					new_MI_learned += last_nominator + nominator_sum/Nsamples;
					if (new_MI_learned > max_MI_learned) {
						max_MI_learned = new_MI_learned;
						max_nominator = nominator_sum/Nsamples;
						selected_feature = i;
					}
				}
			}

		}
		if (max_MI_learned - last_MI_learned < reload_threshold && initial_step != 0) {
#	//		cout << max_MI_learned << endl;
			cout << "WARNING: cannot increase the lower bound, clear everything" << endl;
			initial_step = 0;
			last_nominator = 0.0;
			last_MI_learned = 0.0;
			for (int i = 0; i < Nsamples; ++i) {
				for (int j = 0; j < featureDistribution[0].numSecondStates; ++j) {
					x_matrix[i][j] = featureDistribution[0].secondProbabilityVector[j];					
				}
			}
			for (int i = 0; i < Nfeatures; ++i)
				for (int j = 0; j < Nsamples; ++j)
					for (int k = 0; k < featureDistribution[i].numSecondStates; ++k) {
						int firstValue = featureDistribution[i].firstStateMap[data[j][i]];
						prob_ratio[i][j][k] = probability_con(featureDistribution[i], firstValue ,k);
						memory_store[i][j][k] = 1.0;
					}
			continue;
		}
		feature_set.insert(selected_feature);
		if (initial_step == 0) {
			top_feature = cur_step;
			cout << "haha" << endl;
			for (int j = 0; j < Nsamples; ++j) {
				for (int k = 0; k < featureDistribution[selected_feature].numSecondStates; ++k) {
					int firstValue = featureDistribution[selected_feature].firstStateMap[data[j][selected_feature]];
					x_matrix[j][k] *= prob_ratio[selected_feature][j][k];
				}
			}
		} else {
			last_feature = feature_array[cur_step-1];
			JointProbabilityState distribution = calculateJointProbability(data_inverse[selected_feature], featureMergeArray[last_feature], Nsamples);
			double new_MI_learned = 0.0;
			double nominator_sum = 0.0;
			for (int j = 0; j < Nsamples; ++j) {
				double tem = 0.0;
				for (int k = 0; k < featureDistribution[selected_feature].numSecondStates; ++k) {
					int firstValue = featureDistribution[selected_feature].firstStateMap[data[j][selected_feature]];
					int secondValue = featureDistribution[last_feature].firstStateMap[data[j][last_feature]]+k*featureDistribution[last_feature].numFirstStates;
					x_matrix[j][k] *= memory_store[selected_feature][j][k];
				}
			} 

		}
		
		last_MI_learned = max_MI_learned;
		last_nominator += max_nominator;
		MI_learned[cur_step] = max_MI_learned;
		feature_array[cur_step] = selected_feature;
		++initial_step;
		cout << "step:" << cur_step++ << ": selected feature " << selected_feature << ", MI lower bound " << max_MI_learned << endl;
		
	}
	cout << file_name << endl;	
	for (int i = 0; i < n_features; ++i) {
		if (i != n_features-1) 
			cout << feature_array[i] << " ";
		else
			cout << feature_array[i] << endl;
	}
	for (int i = 0; i < n_features; ++i) {
		if (i != n_features-1) 
			cout << feature_array[i] << ",";
		else
			cout << feature_array[i] << endl;
	}


}
int main()
{
	readData(file_name+".txt", file_name+"_labels.txt");
	feature_selection_discrete_greedy(min(Nfeatures,selected_features));
}
