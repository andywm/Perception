#if !defined(_WIN32) && !defined(_WIN64) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
#define PLATFORM_UNIX 1
#else
#define PLATFORM_UNIX 0
#endif
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

#ifndef NULL
#define NULL 0
#endif
typedef enum {false=0, true=1} bool;
typedef enum {fn_logistic_sigmoid, fn_threshold, fn_tanh} activ_fn;
typedef enum {FAIL=0, SUCCESS=1} result_state;
typedef struct ionode
{
	struct ionode * ptr_next;
	double distance; 
}ionode;
typedef ionode* node;

/*forward declarations*/
bool readfile(const char * path);
void make_random_weights();
void cleanup();
void train_network();
double activation(double val);
double simulate_perceptron(double input[], int limit);
double get_actual();
void reset_input();
bool simulate_input(double input[]);
double normalised_rand();

/*Globals*/
const int MAX_WEIGHT=4;
node prev=NULL, data = NULL, top = NULL;
const char * path;
double MSE=0;
double weights[] = {0,0,0,0};
int dcount=0;
/*configurations*/
const char unixpath[] = "data/dataset.txt";
const char winpath[] = "data\\dataset.txt";
const double LEARNING_RATE = 0.01;
const double TERMINATE_BOUNDARY = 0.0001;
const int MAX_CYCLES = 200;//00;
const double THRESHOLD=0.5; //for activ_fn fn_threshold.
//const activ_fn ACTIVATION_FN = fn_tanh;
/*options are*/
const activ_fn ACTIVATION_FN = fn_threshold;
//const activ_fn ACTIVATION_FN = fn_logistic_sigmoid;
//const activ_fn ACTIVATION_FN = fn_tanh;

/* 
program "Perceptron" Andrew Woodward "andywm", March 2016; last updated April 2016.
*/

int main()
{
	printf("Perceptron Program - Initialising...\n");
	result_state state = SUCCESS;
	if(PLATFORM_UNIX)
	{
		path = unixpath;
	}
	else
	{
		path = winpath;
	}
	printf("Platform is %s.\n", ((PLATFORM_UNIX)? "UNIX" : "WINDOWS"));
	if(readfile(path)==SUCCESS)
	{
		make_random_weights();
		train_network();
	}
	else {state = FAIL;}
	cleanup();
	return state;
}

bool readfile(const char * path)
{
	const int BSIZE = 256;
	char buffer[BSIZE];
	FILE *fp=NULL;
	printf("Attempting to open [%s] ... ", path);
	fp = fopen(path, "r");
	if(fp==NULL) 
	{
		printf("Failed.\n");
		return FAIL;
	}
	else{printf("Done.\n");}

	//set data head.
	data = malloc(sizeof(ionode));
	top = malloc(sizeof(ionode));
	data->distance=0;
	top->distance = 0;
	data->ptr_next=top;
	top->ptr_next = NULL;
	prev = top;
	printf("Reading Data ... ");

	//get rest of data...
	do
	{
		if(fgets(buffer, BSIZE, fp) != NULL)
		{
			top = malloc(sizeof(ionode));
			prev->ptr_next = top;
			//initialise node
			top->distance = strtod(buffer, NULL);
			top->ptr_next=NULL;
			prev=top;
			dcount++;
		}
		else
		{
			if(dcount == 0)
			{
				if(fp!=NULL) fclose(fp);
				printf("Bad data!\n");
			}
	
			break;
		}
	}while(true);
	printf("Done [count:%i]+2. Will cycle for %i\n", dcount, dcount-1);
	if(fp!=NULL) fclose(fp);	

	return SUCCESS;		
}

void cleanup()
{
	while(data!=NULL)
	{
		node data2 = data->ptr_next;
		free(data);
		data=data2;
	}
	data=NULL;
	top=NULL;
	dcount=0;
}

void make_random_weights()
{
	time_t t;
	srand(time(&t));
	weights[0] = normalised_rand();
	weights[1] = normalised_rand();
	weights[2] = normalised_rand();
	weights[3] = normalised_rand();
}

double normalised_rand()
{
	int sign = rand() % 2;
	double rnd = ((double)rand()/(double)RAND_MAX)/2;
	return ((sign==0)? -1 : 1) * rnd;
}

double sum_sq_err;
void train_network()
{
	printf("Now Training Network...\n");
	int iteration, sample=0;
	//For each sample

    int w_lim = 3;
	if(ACTIVATION_FN == fn_threshold) w_lim=4;

    for(iteration=0; iteration < MAX_CYCLES; iteration++)
	{
		reset_input();
        MSE=0; sum_sq_err=0;
		double prediction=0, actual=0, error =0, sample=0;
		double input[3];
        while(simulate_input(input))
		{
			sample++;
			prediction = simulate_perceptron(input, 3);
			actual = get_actual();
			error = actual - prediction;
			//if(iteration+1==CYCLES) printf("%f\n", prediction);
            sum_sq_err+=(error*error);
            int i;
            for(i=0; i<w_lim; i++)
            {
                //wi = wi + d3 * error * learning_rate
                weights[i] += input[2] * error * LEARNING_RATE;
            }
            printf("%f\n", prediction);
		}
        MSE=(sum_sq_err)/(sample+1);
        printf("For iteration %i: MSE of %f\n" ,iteration+1, MSE);
        if(MSE < TERMINATE_BOUNDARY) {printf("Training terminated: MSE(%f)<TerminateBoundary(%f)\n", MSE, TERMINATE_BOUNDARY); break;}
	}
    printf("Training complete...\n");
}

double simulate_perceptron(double input[], int limit)
{
	double net_sum=0;
	int i;
	if(limit ==4){ limit = 3; net_sum = THRESHOLD*weights[3];}//case for threshold...
	for(i=0; i<limit; i++)
	{
		//net_sum = d1 * w1 + d2 * w2 + d3 * w3
		net_sum+= input[i]*weights[i];
	}
	return activation(net_sum);
}
bool simulate_input(double input[])
{
	if(top == NULL || top->ptr_next == NULL || top->ptr_next->ptr_next == NULL || top->ptr_next->ptr_next->ptr_next == NULL) return false; //safe!
	input[0] = (double)(top->distance);
	input[1] = (double)(top->ptr_next->distance);
	input[2] = (double)(top->ptr_next->ptr_next->distance);

	top = top->ptr_next; //incriment top.
}
double get_actual()
{
	if(top == NULL || top->ptr_next == NULL || top->ptr_next->ptr_next == NULL) return -1;
	return (top->ptr_next->ptr_next->distance);
}
void reset_input()
{
	top = data;
}
double activation(double val)
{
	switch(ACTIVATION_FN)
	{
		case fn_logistic_sigmoid:
			return 1 / (1 + pow(M_E, -val));
			/*
			  S(val) = ____1_____
			           1+e^(-val)
			*/
		case fn_threshold:
			return ((val<THRESHOLD)? 0 : 1.0);
		case fn_tanh:
			return tanh(val);
	}
}

//andywm// "Perceptron"  08336 ACW part i.
