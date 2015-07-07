
#include <iostream>

#include "mpi.h"
#include "omp.h"

#ifndef Add_
#define FORTRAN(name) name
#define BLAS(name) name
#define LAPACK(name) name
#else
#define FORTRAN(name) name##_
#define BLAS(name) name##_
#define LAPACK(name) name##_
#endif


extern "C" {

void FORTRAN(dgemm) 
( const char* transA, const char* transB,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* A, const int* lda, 
                       const double* B, const int* ldb, 
  const double* beta,        double* C, const int* ldc );

 }


using namespace std;

int main(int argc, char * argv[]){

  MPI_Init(&argc,&argv);
  int npack1, ne,dummy;

  cin>>npack1;
  cin>>ne;
  cin>>dummy;
  cout<<npack1<<" "<<ne<<endl; 

  double * psi1, *psi2, *matrix;
  double tstart,tstop;


  psi1 = (double*)malloc(npack1*ne*sizeof(double)); 
  psi2 = (double*)malloc(npack1*ne*sizeof(double)); 
  matrix = (double*)malloc(ne*ne*sizeof(double)); 


#pragma omp parallel for
  for(int i=0;i<npack1*ne;++i){
    psi1[i] = 2.0*rand() - 1.0;
    psi2[i] = psi1[i];
  }

  do{

#ifdef CALL_MKL
    tstart=MPI_Wtime();
    double one = 1.0;
    double zero = 0.0;
    FORTRAN(dgemm)("T","N",&ne,&ne,&npack1,&one,psi2,&npack1,psi1,&npack1,&zero,matrix,&ne);

//    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
//        ne, ne, npack1, 1.0, psi2, npack1, psi1, npack1, 0.0, matrix, ne);
    tstop=MPI_Wtime();
    cout<<"Time: "<<tstop-tstart<<"s"<<endl;
#else
    tstart=MPI_Wtime();
#pragma omp parallel firstprivate(npack1,ne)
    {
      int tid = omp_get_thread_num();
      int nthr = omp_get_num_threads();
      int nida = 1;

      {
#pragma omp taskgroup
        {
#pragma omp single nowait
          for(int expe=1;expe<=3;++expe){
            {
              for(int k=1;k<=ne;++k){
#pragma omp task untied firstprivate(k,ne,npack1)
                {
                  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                      k,1, 2*nida , 1.0, &psi2[0], npack1, &psi1[0 + (k-1)*npack1  ], npack1, 0.0, &matrix[0 + (k-1)*ne], k);
                  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                      k,1, npack1 - 2*nida , 2.0, &psi2[2*nida-1], npack1, &psi1[2*nida-1  + (k-1)*npack1], npack1, 1.0, &matrix[0 + (k-1)*ne ], k);
                }
              }
            }
          }
        }

#pragma omp taskgroup
        {
#pragma omp single nowait
          for(int expe=1;expe<=3;++expe){
            for(int k=1; k<=ne; ++k){
#pragma omp task untied firstprivate(k,ne,expe)
              for(int j =k+1;j<=ne;++j){
                matrix[j + (k-1)*ne -1] = matrix[k + (j-1)*ne -1];
              }
            }
          }
        }
      }
    }
    tstop=MPI_Wtime();
    cout<<"Time: "<<tstop-tstart<<"s"<<endl;
#endif


    npack1=-1;
    ne=-1;
    cin>>npack1;
    cin>>ne;
    cin>>dummy;
    if(npack1!=-1 && ne!=-1){
      cout<<npack1<<" "<<ne<<endl;
    } 
  }while(npack1!=-1 && ne!=-1);

  free(matrix);
  free(psi2);
  free(psi1);

  MPI_Finalize();
}
