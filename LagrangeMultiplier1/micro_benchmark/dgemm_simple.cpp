
#include <iostream>
#include <cmath>

#include <stdio.h>
#include <string.h>
#include "mkl.h"
#include "mpi.h"
#include "omp.h"

#ifdef NO_MKL


/*this is for A^T*/
//
#define DGEMM( storeFormat, transA, transB, m, n ,k, alpha, A, lda, B, ldb, beta, C, ldc) \
    do{\
    for(int dgi=0;dgi<m;++dgi){\
      for(int dgj=0;dgj<n;++dgj){\
        double c = beta*((double*)C)[dgi*ldc + dgj];\
        for(int dgk=0;dgk<k;++dgk){\
          c+=alpha*((double*)A)[dgi*lda+dgk]*((double*)B)[dgj*ldb+dgk];\
        }\
        ((double*)C)[dgi*ldc + dgj]=c;\
      }\
    }\
    }while(0)

#else
#define DGEMM( storeFormat, transA, transB, m, n ,k, alpha, A, lda, B, ldb, beta, C, ldc) \
    cblas_dgemm( storeFormat, transA, transB, m, n ,k, alpha, A, lda, B, ldb, beta, C, ldc)
      
#endif


//#define VERBOSE

using namespace std;

int main(int argc, char * argv[]){

  MPI_Init(&argc,&argv);
  int npack1, ne,dummy;

  cin>>npack1;
  cin>>ne;
  cin>>dummy;
  cout<<npack1<<" "<<ne<<endl; 

  double * psi1, *psi2, *matrix;
  double * psi3, *psi4, *matrix2;
  double * psi5, *psi6, *matrix3;
  double tstart,tstop;


  psi1 = (double*)mkl_malloc(npack1*ne*sizeof(double),64); 
  psi2 = (double*)mkl_malloc(npack1*ne*sizeof(double),64); 
  matrix = (double*)mkl_malloc(ne*ne*sizeof(double),64); 

  psi3 = (double*)mkl_malloc(npack1*ne*sizeof(double),64); 
  psi4 = (double*)mkl_malloc(npack1*ne*sizeof(double),64); 
  matrix2 = (double*)mkl_malloc(ne*ne*sizeof(double),64); 

  psi5 = (double*)mkl_malloc(npack1*ne*sizeof(double),64); 
  psi6 = (double*)mkl_malloc(npack1*ne*sizeof(double),64); 
  matrix3 = (double*)mkl_malloc(ne*ne*sizeof(double),64);

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
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
        ne, ne, npack1, 1.0, psi4, npack1, psi3, npack1, 0.0, matrix2, ne);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
        ne, ne, npack1, 1.0, psi6, npack1, psi5, npack1, 0.0, matrix3, ne);
    tstop=MPI_Wtime();
    cout<<"Time: "<<tstop-tstart<<"s"<<endl;
#else


#if 0

    tstart=MPI_Wtime();
#pragma omp parallel firstprivate(npack1,ne)
    {
      int tid = omp_get_thread_num();
      int nthr = omp_get_num_threads();
      int nida = 1;

      {
#pragma omp taskgroup
        {
                //this is the width of a block
                int bw = 1;
                int numStepw = std::ceil((double)ne/(double)bw);
//i is col index, j is row index
#pragma omp single nowait
              for(int ib=1;ib<=numStepw;++ib){
                  int i = (ib-1)*bw+1;
                  int endi = min((ib)*bw,ne)+1;
                  int tbw = endi-i;
                  //this is the height of a block
                  int bh = endi;
                  //int bh = 60;
                  int numSteph = std::ceil((double)ne/(double)bh);
                for(int jb=1;jb<=numSteph;++jb){
                  int j = (jb-1)*bh+1;
                  int endj = min((jb)*bh,ne)+1;
                  int tbh = endj-j;
                  if(endj<=endi){
#pragma omp task untied firstprivate(ne,npack1,i,j,tbh,tbw)
                    {
//      int tid = omp_get_thread_num();
//printf("T%d doing block C(%d..%d,%d..%d) = A(%d..%d,%d..%d)*B(%d..%d,%d..%d)\n",tid,j,j+tbh-1,i,i+tbw-1,j,j+tbh-1,1,npack1,1,npack1,i,i+tbw-1);
                      //start from the jth row
                      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                          tbh,tbw, 2*nida , 1.0, &psi2[0+(j-1)*npack1], npack1, &psi1[0 + (i-1)*npack1 ], npack1, 0.0, &matrix[0 + (i-1)*ne + (j-1)], ne);
                      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                          tbh,tbw, npack1 - 2*nida , 2.0, &psi2[2*nida-1 + (j-1)*npack1], npack1, &psi1[2*nida-1  + (i-1)*npack1], npack1, 1.0, &matrix[0 + (i-1)*ne + (j-1) ], ne);
                    }
#pragma omp task untied firstprivate(ne,npack1,i,j,tbh,tbw)
                    {
//      int tid = omp_get_thread_num();
//printf("T%d doing block C2(%d..%d,%d..%d) = A2(%d..%d,%d..%d)*B2(%d..%d,%d..%d)\n",tid,j,j+tbh-1,i,i+tbw-1,j,j+tbh-1,1,npack1,1,npack1,i,i+tbw-1);
                      //start from the jth row
                      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                          tbh,tbw, 2*nida , 1.0, &psi4[0+(j-1)*npack1], npack1, &psi3[0 + (i-1)*npack1 ], npack1, 0.0, &matrix2[0 + (i-1)*ne + (j-1)], ne);
                      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                          tbh,tbw, npack1 - 2*nida , 2.0, &psi4[2*nida-1 + (j-1)*npack1], npack1, &psi3[2*nida-1  + (i-1)*npack1], npack1, 1.0, &matrix2[0 + (i-1)*ne + (j-1) ], ne);
                    }
#pragma omp task untied firstprivate(ne,npack1,i,j,tbh,tbw)
                    {
//      int tid = omp_get_thread_num();
//printf("T%d doing block C2(%d..%d,%d..%d) = A2(%d..%d,%d..%d)*B2(%d..%d,%d..%d)\n",tid,j,j+tbh-1,i,i+tbw-1,j,j+tbh-1,1,npack1,1,npack1,i,i+tbw-1);
                      //start from the jth row
                      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                          tbh,tbw, 2*nida , 1.0, &psi6[0+(j-1)*npack1], npack1, &psi5[0 + (i-1)*npack1 ], npack1, 0.0, &matrix3[0 + (i-1)*ne + (j-1)], ne);
                      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                          tbh,tbw, npack1 - 2*nida , 2.0, &psi6[2*nida-1 + (j-1)*npack1], npack1, &psi5[2*nida-1  + (i-1)*npack1], npack1, 1.0, &matrix3[0 + (i-1)*ne + (j-1) ], ne);
                    }
                }
              }
            }
        }

#pragma omp taskgroup
        {
#pragma omp single nowait
            for(int k=1; k<=ne; ++k){
#pragma omp task untied firstprivate(k,ne)
              for(int j =k+1;j<=ne;++j){
                matrix[j + (k-1)*ne -1] = matrix[k + (j-1)*ne -1];
              }
            }
            for(int k=1; k<=ne; ++k){
#pragma omp task untied firstprivate(k,ne)
              for(int j =k+1;j<=ne;++j){
                matrix2[j + (k-1)*ne -1] = matrix2[k + (j-1)*ne -1];
              }
            }
            for(int k=1; k<=ne; ++k){
#pragma omp task untied firstprivate(k,ne)
              for(int j =k+1;j<=ne;++j){
                matrix3[j + (k-1)*ne -1] = matrix3[k + (j-1)*ne -1];
              }
            }
        }
      }
    }
    tstop=MPI_Wtime();
    cout<<"Time: "<<tstop-tstart<<"s"<<endl;




#else
      int nida = 1;
int BK = ((npack1-2*nida)/1);//omp_get_max_threads());
int BW = 4;//ne;//ne/8;
int BH = 4;//ne;//ne/8;

                printf("BK=%d BW=%d BH=%d\n",BK,BW,BH);

                //this is the width of a block
                int bk = BK;
                int numStepk = std::ceil((double)(npack1-2*nida)/(double)bk);


  double * matrixDup = (double*)mkl_malloc(numStepk*ne*ne*sizeof(double),64); 
  std::fill(matrixDup,matrixDup+numStepk*ne*ne,0.0);
  double * matrix2Dup = (double*)mkl_malloc(numStepk*ne*ne*sizeof(double),64); 
  std::fill(matrix2Dup,matrix2Dup+numStepk*ne*ne,0.0);
  double * matrix3Dup = (double*)mkl_malloc(numStepk*ne*ne*sizeof(double),64); 
  std::fill(matrix3Dup,matrix3Dup+numStepk*ne*ne,0.0);

    tstart=MPI_Wtime();
#pragma omp parallel firstprivate(npack1,ne,bk,numStepk,nida,BK,BW,BH)
    {
//                        double matrix_private[tbh*tbw];
//                        double matrix_private2[tbh*tbw];
//                        double matrix_private3[tbh*tbw];


      int tid = omp_get_thread_num();
      int nthr = omp_get_num_threads();

      {
#pragma omp taskgroup
        {


                int bw = BW;
                int numStepw = std::ceil((double)ne/(double)bw);
//i is col index, j is row index
#pragma omp single nowait
              for(int ib=1;ib<=numStepw;++ib){
                  int i = (ib-1)*bw+1;
                  int endi = min((ib)*bw,ne)+1;
                  int tbw = endi-i;
                  //this is the height of a block
                  int bh = BH;//endi;
                  //int bh = 60;
                  int numSteph = std::ceil((double)ne/(double)bh);
                for(int jb=1;jb<=numSteph;++jb){
                  int j = (jb-1)*bh+1;
                  int endj = min((jb)*bh,ne)+1;
                  int tbh = endj-j;
                  if(endj<=endi){

#pragma omp task untied firstprivate(bk,ne,npack1,i,j,tbh,tbw) depend(out: matrix[0 + (i-1)*ne + (j-1) ])
                      {
#ifdef VERBOSE
                                int tid = omp_get_thread_num();
                          printf("T%d doing FIRST block C(%d..%d,%d..%d) = A(%d..%d,%d..%d)*B(%d..%d,%d..%d)\n",tid,j,j+tbh-1,i,i+tbw-1,j,j+tbh-1,1,2*nida,1,2*nida,i,i+tbw-1);
#endif
                        //start from the jth row
                        DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, 
                            tbh,tbw, 2*nida , 1.0, &psi2[0+(j-1)*npack1], npack1, &psi1[0 + (i-1)*npack1 ], npack1, 0.0, &matrix[0 + (i-1)*ne + (j-1) ],ne);
                      }

                      for(int kb=1;kb<=numStepk;++kb){

                        int k = (kb-1)*bk+1;
                        int endk = min((kb)*bk,npack1-2*nida)+1;
                        int tbk = endk-k;

#pragma omp task untied firstprivate(bk,ne,npack1,i,j,tbh,tbw,k,tbk) depend(inout: matrixDup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ])
                        {
#ifdef VERBOSE
                                int tid = omp_get_thread_num();
                          printf("T%d doing block C(%d..%d,%d..%d)_{%d} = A(%d..%d,%d..%d)*B(%d..%d,%d..%d)\n",tid,j,j+tbh-1,i,i+tbw-1,kb,j,j+tbh-1,k,k+tbk-1,k,k+tbk-1,i,i+tbw-1);
#endif
                          //
                          //start from the jth row
                          DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, 
                              tbh,tbw, tbk , 2.0, &psi2[2*nida-1 + k-1 + (j-1)*npack1], npack1, &psi1[2*nida-1 + k-1  + (i-1)*npack1], npack1, 1.0, &matrixDup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ], ne);
                        }
                      }

                      for(int kb=1;kb<=numStepk;++kb){

                        int k = (kb-1)*bk+1;
                        int endk = min((kb)*bk,npack1-2*nida)+1;
                        int tbk = endk-k;

#pragma omp task untied firstprivate(bk,ne,npack1,i,j,tbh,tbw,k,tbk) depend(in: matrixDup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ]) depend(inout: matrix[0 + (i-1)*ne + (j-1) ])
{
#ifdef VERBOSE
                          int tid = omp_get_thread_num();
                          printf("T%d reducing block C(%d..%d,%d..%d)_{%d}\n",tid,j,j+tbh-1,i,i+tbw-1,kb);
#endif
#pragma omp critical
                        cblas_daxpy(tbh*tbw,1.0,&matrixDup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ],ne,&matrix[0 + (i-1)*ne + (j-1) ],ne);
} 
                     }









#pragma omp task untied firstprivate(ne,npack1,i,j,tbh,tbw) depend(out: matrix2[0 + (i-1)*ne + (j-1) ])
                      {
#ifdef VERBOSE
                                int tid = omp_get_thread_num();
                          printf("T%d doing FIRST block C(%d..%d,%d..%d) = A(%d..%d,%d..%d)*B(%d..%d,%d..%d)\n",tid,j,j+tbh-1,i,i+tbw-1,j,j+tbh-1,1,2*nida,1,2*nida,i,i+tbw-1);
#endif
                        //start from the jth row
                        DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, 
                            tbh,tbw, 2*nida , 1.0, &psi4[0+(j-1)*npack1], npack1, &psi3[0 + (i-1)*npack1 ], npack1, 0.0, &matrix2[0 + (i-1)*ne + (j-1) ],ne);
                      }

                      for(int kb=1;kb<=numStepk;++kb){

                        int k = (kb-1)*bk+1;
                        int endk = min((kb)*bk,npack1-2*nida)+1;
                        int tbk = endk-k;

#pragma omp task untied firstprivate(ne,npack1,i,j,tbh,tbw,k,tbk) depend(inout: matrix2Dup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ])
                        {
#ifdef VERBOSE
                                int tid = omp_get_thread_num();
                          printf("T%d doing block C(%d..%d,%d..%d)_{%d} = A(%d..%d,%d..%d)*B(%d..%d,%d..%d)\n",tid,j,j+tbh-1,i,i+tbw-1,kb,j,j+tbh-1,k,k+tbk-1,k,k+tbk-1,i,i+tbw-1);
#endif
                          //
                          //start from the jth row
                          DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, 
                              tbh,tbw, tbk , 2.0, &psi4[2*nida-1 + k-1 + (j-1)*npack1], npack1, &psi3[2*nida-1 + k-1  + (i-1)*npack1], npack1, 1.0, &matrix2Dup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ], ne);
                        }
                      }

                      for(int kb=1;kb<=numStepk;++kb){

                        int k = (kb-1)*bk+1;
                        int endk = min((kb)*bk,npack1-2*nida)+1;
                        int tbk = endk-k;

#pragma omp task untied firstprivate(ne,npack1,i,j,tbh,tbw,k,tbk) depend(in: matrix2Dup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ]) depend(inout: matrix2[0 + (i-1)*ne + (j-1) ])
{
#ifdef VERBOSE
                          int tid = omp_get_thread_num();
                          printf("T%d reducing block C(%d..%d,%d..%d)_{%d}\n",tid,j,j+tbh-1,i,i+tbw-1,kb);
#endif
#pragma omp critical
                        cblas_daxpy(tbh*tbw,1.0,&matrix2Dup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ],ne,&matrix2[0 + (i-1)*ne + (j-1) ],ne);
} 
                     }




#pragma omp task untied firstprivate(ne,npack1,i,j,tbh,tbw) depend(out: matrix3[0 + (i-1)*ne + (j-1) ])
                      {
#ifdef VERBOSE
                                int tid = omp_get_thread_num();
                          printf("T%d doing FIRST block C(%d..%d,%d..%d) = A(%d..%d,%d..%d)*B(%d..%d,%d..%d)\n",tid,j,j+tbh-1,i,i+tbw-1,j,j+tbh-1,1,2*nida,1,2*nida,i,i+tbw-1);
#endif
                        //start from the jth row
                        DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, 
                            tbh,tbw, 2*nida , 1.0, &psi6[0+(j-1)*npack1], npack1, &psi5[0 + (i-1)*npack1 ], npack1, 0.0, &matrix3[0 + (i-1)*ne + (j-1) ],ne);
                      }

                      for(int kb=1;kb<=numStepk;++kb){

                        int k = (kb-1)*bk+1;
                        int endk = min((kb)*bk,npack1-2*nida)+1;
                        int tbk = endk-k;

#pragma omp task untied firstprivate(ne,npack1,i,j,tbh,tbw,k,tbk) depend(inout: matrix3Dup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ])
                        {
#ifdef VERBOSE
                                int tid = omp_get_thread_num();
                          printf("T%d doing block C(%d..%d,%d..%d)_{%d} = A(%d..%d,%d..%d)*B(%d..%d,%d..%d)\n",tid,j,j+tbh-1,i,i+tbw-1,kb,j,j+tbh-1,k,k+tbk-1,k,k+tbk-1,i,i+tbw-1);
#endif
                          //
                          //start from the jth row
                          DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, 
                              tbh,tbw, tbk , 2.0, &psi6[2*nida-1 + k-1 + (j-1)*npack1], npack1, &psi5[2*nida-1 + k-1  + (i-1)*npack1], npack1, 1.0, &matrix3Dup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ], ne);
                        }
                      }

                      for(int kb=1;kb<=numStepk;++kb){

                        int k = (kb-1)*bk+1;
                        int endk = min((kb)*bk,npack1-2*nida)+1;
                        int tbk = endk-k;

#pragma omp task untied firstprivate(ne,npack1,i,j,tbh,tbw,k,tbk) depend(in: matrix3Dup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ]) depend(inout: matrix3[0 + (i-1)*ne + (j-1) ])
{
#ifdef VERBOSE
                          int tid = omp_get_thread_num();
                          printf("T%d reducing block C(%d..%d,%d..%d)_{%d}\n",tid,j,j+tbh-1,i,i+tbw-1,kb);
#endif
#pragma omp critical
                        cblas_daxpy(tbh*tbw,1.0,&matrix3Dup[0 + (i-1)*ne + (j-1) + (kb-1)*ne*ne ],ne,&matrix3[0 + (i-1)*ne + (j-1) ],ne);
} 
                     }




                }
              }
            }
        }

#pragma omp taskgroup
        {
#pragma omp single nowait
            for(int k=1; k<=ne; ++k){
#pragma omp task untied firstprivate(k,ne)
              for(int j =k+1;j<=ne;++j){
                matrix[j + (k-1)*ne -1] = matrix[k + (j-1)*ne -1];
              }
            }
            for(int k=1; k<=ne; ++k){
#pragma omp task untied firstprivate(k,ne)
              for(int j =k+1;j<=ne;++j){
                matrix2[j + (k-1)*ne -1] = matrix2[k + (j-1)*ne -1];
              }
            }
            for(int k=1; k<=ne; ++k){
#pragma omp task untied firstprivate(k,ne)
              for(int j =k+1;j<=ne;++j){
                matrix3[j + (k-1)*ne -1] = matrix3[k + (j-1)*ne -1];
              }
            }
        }
      }
    }
    tstop=MPI_Wtime();
    cout<<"Time: "<<tstop-tstart<<"s"<<endl;

  mkl_free(matrix3Dup);
  mkl_free(matrix2Dup);
  mkl_free(matrixDup);

#endif










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

  mkl_free(matrix2);
  mkl_free(psi3);
  mkl_free(psi4);

  mkl_free(matrix3);
  mkl_free(psi5);
  mkl_free(psi6);

  MPI_Finalize();
}
