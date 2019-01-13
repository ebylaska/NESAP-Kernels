
#include <iostream>

#include "mkl.h"
#include "mpi.h"
#include "omp.h"

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


  psi1 = (double*)mkl_malloc(npack1*ne*sizeof(double),64); 
  psi2 = (double*)mkl_malloc(npack1*ne*sizeof(double),64); 
  matrix = (double*)mkl_malloc(ne*ne*sizeof(double),64); 


#pragma omp parallel for
  for(int i=0;i<npack1*ne;++i){
    psi1[i] = 2.0*rand() - 1.0;
    psi2[i] = psi1[i];
  }

  do{

#ifdef CALL_MKL
    tstart=MPI_Wtime();
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
        ne, ne, npack1, 1.0, psi2, npack1, psi1, npack1, 0.0, matrix, ne);
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

  mkl_free(matrix);
  mkl_free(psi2);
  mkl_free(psi1);

  MPI_Finalize();
}
