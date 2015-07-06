
#include <iostream>

#include "mpi.h"
#include "blis.h"
#include "omp.h"


using namespace std;

int main(int argc, char * argv[]){

  MPI_Init(&argc,&argv);
  int npack1, ne,dummy;

  cin>>npack1;
  cin>>ne;
  cin>>dummy;
  cout<<npack1<<" "<<ne<<endl; 

	obj_t psi1, psi2, matrix;
	obj_t alpha, beta;
	dim_t b_npack1, b_ne;
	num_t dt;
	trans_t  transa;
	trans_t  transb;

	bli_init();

	dt = BLIS_DOUBLE;

	transa = BLIS_TRANSPOSE;
	transb = BLIS_NO_TRANSPOSE;
  
  b_npack1 = (dim_t)npack1;
  b_ne = (dim_t)ne;

  bli_obj_create( dt, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt, 1, 1, 0, 0, &beta );

		bli_obj_create( dt, b_npack1, b_ne, 0, 0, &psi1 );
		bli_obj_create( dt, b_npack1, b_ne, 0, 0, &psi2 );
		bli_obj_create( dt, b_ne, b_ne, 0, 0, &matrix );

		bli_randm( &psi1 );
		bli_copym( &psi1, &psi2 );

		bli_obj_set_conjtrans( transa, psi1 );
		bli_obj_set_conjtrans( transb, psi2 );

		bli_setsc(  1.0, 0.0, &alpha );
		bli_setsc( 0.0, 0.0, &beta );
  double tstart,tstop;

  do{
    tstart=MPI_Wtime();
			bli_gemm( &alpha,
			          &psi1,
			          &psi2,
			          &beta,
			          &matrix );

    tstop=MPI_Wtime();
    cout<<"Time: "<<tstop-tstart<<"s"<<endl;

    npack1=-1;
    ne=-1;
    cin>>npack1;
    cin>>ne;
    cin>>dummy;
    if(npack1!=-1 && ne!=-1){
      cout<<npack1<<" "<<ne<<endl;
    } 
  }while(npack1!=-1 && ne!=-1);


		bli_obj_free( &alpha );
		bli_obj_free( &beta );

		bli_obj_free( &psi1 );
		bli_obj_free( &psi2 );
		bli_obj_free( &matrix );

	bli_finalize();

  MPI_Finalize();
}
