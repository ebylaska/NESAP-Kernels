
#include <iostream>
#include <cmath>

#include <stdio.h>
#include <string.h>

#ifdef MKL
#include "mkl.h"
#else
#include "cblas.h" 
#endif

#include "mpi.h"
#include <omp.h>

#include <stdlib.h>

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

  int num_tests = 10;
  MPI_Init(&argc,&argv);

  int mpi_np =0;
  int mpi_iam =0;
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_np);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_iam);

  int npack, ne,dummy,onpack,npack1;

  if(mpi_iam==0){
    cin>>npack;
    cin>>ne;
    cin>>dummy;
  }

  MPI_Bcast(&npack,sizeof(int),MPI_BYTE,0,MPI_COMM_WORLD);
  MPI_Bcast(&ne,sizeof(int),MPI_BYTE,0,MPI_COMM_WORLD);

  onpack = npack;
  npack = floor((double)npack/(double)mpi_np);
  if(mpi_iam==mpi_np-1){
    npack = onpack - mpi_iam*npack;
  }
  npack1 = 2*npack;
  if(mpi_iam==0){
    cout<<onpack<<" "<<ne<<endl; 
  }

  double * psi1, *psi2, *matrix;
  double * psi3, *psi4, *matrix2;
  double * psi5, *psi6, *matrix3;
  double tstart,tstop;

  do{

#ifdef MKL
    psi1 = (double*)mkl_malloc(npack1*ne*sizeof(double),64); 
    psi2 = (double*)mkl_malloc(npack1*ne*sizeof(double),64); 
    psi3 = (double*)mkl_malloc(npack1*ne*sizeof(double),64); 
    matrix = (double*)mkl_malloc(ne*ne*sizeof(double),64); 
    matrix2 = (double*)mkl_malloc(ne*ne*sizeof(double),64); 
    matrix3 = (double*)mkl_malloc(ne*ne*sizeof(double),64);
#else
    psi1 = (double*)malloc(npack1*ne*sizeof(double)); 
    psi2 = (double*)malloc(npack1*ne*sizeof(double)); 
    psi3 = (double*)malloc(npack1*ne*sizeof(double)); 
    matrix = (double*)malloc(ne*ne*sizeof(double)); 
    matrix2 = (double*)malloc(ne*ne*sizeof(double)); 
    matrix3 = (double*)malloc(ne*ne*sizeof(double));
#endif

#pragma omp parallel for
    for(int i=0;i<npack1*ne;++i){
      psi1[i] = 2.0*rand() - 1.0;
      psi2[i] = 2.0*rand() - 1.0;
      psi3[i] = 2.0*rand() - 1.0;
    }




    int nida = 1;
#ifdef NESTED
    int np = omp_get_max_threads();
    int divk = np;
    int M=ne;
    int N=ne;
    int K=npack1-2*nida;
#ifdef MKL
    double * matrixDup =  (double*)mkl_malloc(divk*M*N*sizeof(double),64); 
    double * matrix2Dup = (double*)mkl_malloc(divk*M*N*sizeof(double),64); 
    double * matrix3Dup = (double*)mkl_malloc(divk*M*N*sizeof(double),64); 
#else
    double * matrixDup =  (double*)malloc(divk*M*N*sizeof(double)); 
    double * matrix2Dup = (double*)malloc(divk*M*N*sizeof(double)); 
    double * matrix3Dup = (double*)malloc(divk*M*N*sizeof(double)); 
#endif
    //  std::fill(matrixDup,matrixDup+divk*M*N,0.0);
    //  std::fill(matrix2Dup,matrix2Dup+divk*M*N,0.0);
    //  std::fill(matrix3Dup,matrix3Dup+divk*M*N,0.0);




    tstart=MPI_Wtime();

    for(int test=0;test<num_tests;++test){
#pragma omp parallel firstprivate(M,N,K,divk,np,npack1,ne) 
      {

        int iam = omp_get_thread_num();

        int bi = M;
        int bj = N;
        int bk = floor((double)K/(double)divk);
        int offsetk = iam*bk;

        if(iam==np-1){
          bk = K-offsetk;
        }

        
          //printf("T%d \n",iam);



#ifdef MY_NESTED

        //#pragma omp declare reduction (merge : double * : cblas_daxpy(bi*bj,1.0,&omp_in[0 + iam*M*N ],1,&omp_out[0],1))

#pragma omp parallel firstprivate(iam,bi,bj,bk,offsetk,npack1) 
        {
          int t_iam = omp_get_thread_num();
          int t_np = omp_get_num_threads();

          omp_set_num_threads(1);

          int t_divnp = std::ceil(sqrt(t_np));

          int numCol = t_divnp;
          int numRow = std::ceil((double)t_np / (double)numCol);


          int myRow = t_iam/numCol;
          int myCol = t_iam%numCol;

          int bbi = floor((double)bi/(double)numRow);
          int offseti = myRow*bbi;
          if(t_iam==t_np-1){
            bbi = bi - offseti;
          }

          int bbj = floor((double)bj/(double)numCol);
          int offsetj = myCol*bbj;
          if(t_iam==t_np-1){
            bbj = bj - offsetj;
          }

#ifdef VERBOSE
          printf("T%d_%d numRow=%d numCol=%d\n",iam,t_iam,numRow,numCol);
          printf("T%d_%d myRow=%d myCol=%d\n",iam,t_iam,myRow,myCol);
          printf("offsetk=%d bk=%d np2=%d\n",offsetk,bk,t_np);
          printf("offseti=%d bbi=%d\n",offseti,bbi);
          printf("offsetj=%d bbj=%d\n",offsetj,bbj);
          printf("T%d_%d doing block C(%d..%d,%d..%d)_{%d} = A(%d..%d,%d..%d)*B(%d..%d,%d..%d)\n",iam,t_iam,offseti,offseti+bbi-1,offsetj,offsetj+bbj-1,iam, offseti,offseti+bbi-1,2*nida + offsetk, 2*nida + offsetk + bk-1, 2*nida + offsetk, 2*nida + offsetk + bk-1, offsetj, offsetj+bbj-1);
#endif
          if(mpi_iam==0 && iam==0){
            DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bbi,bbj,2*nida ,1.0, &psi2[offseti], npack1, &psi2[offsetj], npack1, 0.0, &matrix[0 + offseti*ne+offsetj], M);
          }
          if(mpi_iam==0 && iam==0){
            DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bbi,bbj,2*nida ,1.0, &psi2[offseti], npack1, &psi1[offsetj], npack1, 0.0, &matrix2[0 + offseti*ne+offsetj], M);
          }
          if(mpi_iam==0 && iam==0){
            DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bbi,bbj,2*nida ,1.0, &psi1[offseti], npack1, &psi1[offsetj], npack1, 0.0, &matrix3[0 + offseti*ne+offsetj], M);
          }

#pragma omp barrier

          //psi2 x psi2
          DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bbi,bbj,bk , 2.0, &psi2[2*nida + offsetk + offseti], npack1, &psi2[2*nida + offsetk +offsetj], npack1, 0.0, &matrixDup[0 + iam*M*N + offseti*ne+offsetj], M);
          //psi2 x psi1
          DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bbi,bbj,bk , 2.0, &psi2[2*nida + offsetk + offseti], npack1, &psi1[2*nida + offsetk +offsetj], npack1, 0.0, &matrix2Dup[0 + iam*M*N + offseti*ne+offsetj], M);
          //psi1 x psi1
          DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bbi,bbj,bk , 2.0, &psi1[2*nida + offsetk + offseti], npack1, &psi1[2*nida + offsetk +offsetj], npack1, 0.0, &matrix3Dup[0 + iam*M*N + offseti*ne+offsetj], M);
        }

        //reduce contribution
#pragma omp critical
        {
          cblas_daxpy(bi*bj,1.0,&matrixDup[0 + iam*M*N ],1,&matrix[0],1);
        }
        //reduce contribution
#pragma omp critical
        {
          cblas_daxpy(bi*bj,1.0,&matrix2Dup[0 + iam*M*N ],1,&matrix2[0],1);
        }
        //reduce contribution
#pragma omp critical
        {
          cblas_daxpy(bi*bj,1.0,&matrix3Dup[0 + iam*M*N ],1,&matrix3[0],1);
        }
#else
        if(mpi_iam==0 && iam==0){
          DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bi,bj,2*nida , 1.0, &psi2[0], npack1, &psi2[0], npack1, 0.0, &matrix[0], M);
        }
        if(mpi_iam==0 && iam==0){
          DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bi,bj,2*nida , 1.0, &psi2[0], npack1, &psi1[0], npack1, 0.0, &matrix2[0], M);
        }
        if(mpi_iam==0 && iam==0){
          DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bi,bj,2*nida , 1.0, &psi1[0], npack1, &psi1[0], npack1, 0.0, &matrix3[0], M);
        }
#pragma omp barrier



        //psi2 x psi2
        DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bi,bj,bk , 2.0, &psi2[2*nida + offsetk], npack1, &psi2[2*nida + offsetk], npack1, 0.0, &matrixDup[0 + iam*M*N ], M);

        //reduce contribution
#pragma omp critical
        {
          cblas_daxpy(bi*bj,1.0,&matrixDup[0 + iam*M*N ],1,&matrix[0],1);
        }

        //psi2 x psi1
        DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bi,bj,bk , 2.0, &psi2[2*nida + offsetk], npack1, &psi1[2*nida + offsetk], npack1, 0.0, &matrix2Dup[0 + iam*M*N ], M);
        //reduce contribution
#pragma omp critical
        {
          cblas_daxpy(bi*bj,1.0,&matrix2Dup[0 + iam*M*N ],1,&matrix2[0],1);
        }

        //psi1 x psi1
        DGEMM(CblasColMajor, CblasTrans, CblasNoTrans, bi,bj,bk , 2.0, &psi1[2*nida + offsetk], npack1, &psi1[2*nida + offsetk], npack1, 0.0, &matrix3Dup[0 + iam*M*N ], M);

        //reduce contribution
#pragma omp critical
        {
          cblas_daxpy(bi*bj,1.0,&matrix3Dup[0 + iam*M*N ],1,&matrix3[0],1);
        }
#endif
      }

      if(mpi_iam==0){
        MPI_Reduce(MPI_IN_PLACE,(void*)&matrix[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(MPI_IN_PLACE,(void*)&matrix2[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(MPI_IN_PLACE,(void*)&matrix3[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      }
      else{
        MPI_Reduce((void*)&matrix[0], (void*)&matrix[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce((void*)&matrix2[0],(void*)&matrix2[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce((void*)&matrix3[0],(void*)&matrix3[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      }

    }
    tstop=MPI_Wtime();

#ifdef MKL
    mkl_free(matrix3Dup);
    mkl_free(matrix2Dup);
    mkl_free(matrixDup);
#else
    free(matrix3Dup);
    free(matrix2Dup);
    free(matrixDup);
#endif


#else
    tstart=MPI_Wtime();
    for(int test=0;test<num_tests;++test){
      // psi2 x psi2 
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
          ne, ne, npack1-2*nida, 2.0, &psi2[2*nida], npack1, &psi2[2*nida], npack1, 0.0, matrix, ne);
      if(mpi_iam==0){
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
            ne, ne, 2*nida, 1.0, psi2, npack1, psi2, npack1, 1.0, matrix, ne);
      }
      // psi2 x psi1
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
          ne, ne, npack1-2*nida, 2.0, &psi2[2*nida], npack1, &psi1[2*nida], npack1, 0.0, matrix2, ne);

      if(mpi_iam==0){
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
            ne, ne, 2*nida, 1.0, psi2, npack1, psi1, npack1, 1.0, matrix2, ne);
      }
      // psi1 x psi1 
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
          ne, ne, npack1-2*nida, 2.0, &psi1[2*nida], npack1, &psi1[2*nida], npack1, 0.0, matrix3, ne);

      if(mpi_iam==0){
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
            ne, ne, 2*nida, 1.0, psi1, npack1, psi1, npack1, 1.0, matrix3, ne);
      }

      if(mpi_iam==0){
        MPI_Reduce(MPI_IN_PLACE,(void*)&matrix[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(MPI_IN_PLACE,(void*)&matrix2[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(MPI_IN_PLACE,(void*)&matrix3[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      }
      else{
        MPI_Reduce((void*)&matrix[0], (void*)&matrix[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce((void*)&matrix2[0],(void*)&matrix2[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce((void*)&matrix3[0],(void*)&matrix3[0],ne*ne,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      }
    }
    tstop=MPI_Wtime();
#endif
    if(mpi_iam==0){
      cout<<"Time: "<<tstop-tstart<<"s"<<endl;
    }

#ifdef MKL
    mkl_free(psi2);
    mkl_free(psi1);
    mkl_free(psi3);

    mkl_free(matrix);
    mkl_free(matrix2);
    mkl_free(matrix3);
#else
    free(psi2);
    free(psi1);
    free(psi3);

    free(matrix);
    free(matrix2);
    free(matrix3);
#endif



    npack=-1;
    ne=-1;

    if(mpi_iam==0){
      cin>>npack;
      cin>>ne;
      cin>>dummy;
    }

    MPI_Bcast(&npack,sizeof(int),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast(&ne,sizeof(int),MPI_BYTE,0,MPI_COMM_WORLD);




    if(npack!=-1 && ne!=-1){
      onpack = npack;
      npack = floor((double)npack/(double)mpi_np);
      if(mpi_iam==mpi_np-1){
        npack = onpack - mpi_iam*npack;
      }
      npack1 = 2*npack;
      if(mpi_iam==0){
        cout<<onpack<<" "<<ne<<endl; 
      }
    } 
  }while(npack!=-1 && ne!=-1);

  MPI_Finalize();
}
