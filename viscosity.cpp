
#include "viscosity.h"

//  @brief Fortran viscosity kernel.
//  @author Wayne Gaudin
//  @details Calculates an artificial viscosity using the Wilkin's method to
//  smooth out shock front and prevent oscillations around discontinuities.
//  Only cells in compression will have a non-zero value.

void viscosity_kernel(int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double*>& celldx,
  Kokkos::View<double*>& celldy,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& pressure,
  Kokkos::View<double**>& viscosity,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& yvel0) {

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min, y_min}, {y_min, y_max});
  Kokkos::parallel_for("viscosity", policy, KOKKOS_LAMBDA(const int j, const int k) {

    double ugrad = (xvel0(j+1,k  )+xvel0(j+1,k+1))-(xvel0(j  ,k  )+xvel0(j  ,k+1));

    double vgrad = (yvel0(j  ,k+1)+yvel0(j+1,k+1))-(yvel0(j  ,k  )+yvel0(j+1,k  ));

    double div = (celldx(j)*(ugrad)+  celldy(k)*(vgrad));

    double strain2 = 0.5*(xvel0(j,  k+1) + xvel0(j+1,k+1)-xvel0(j  ,k  )-xvel0(j+1,k  ))/celldy(k) 
      + 0.5*(yvel0(j+1,k  ) + yvel0(j+1,k+1)-yvel0(j  ,k  )-yvel0(j  ,k+1))/celldx(j);

    double pgradx=(pressure(j+1,k)-pressure(j-1,k))/(celldx(j)+celldx(j+1));
    double pgrady=(pressure(j,k+1)-pressure(j,k-1))/(celldy(k)+celldy(k+1));

    double pgradx2 = pgradx*pgradx;
    double pgrady2 = pgrady*pgrady;

    double limiter = ((0.5*(ugrad)/celldx(j))*pgradx2+(0.5*(vgrad)/celldy(k))*pgrady2+strain2*pgradx*pgrady)
      /std::max(pgradx2+pgrady2,1.0e-16);

    if ((limiter > 0.0) || (div >= 0.0)) {
      viscosity(j,k) = 0.0;
    } else {
      double dirx=1.0;
      if (pgradx < 0.0) dirx=-1.0;
      pgradx = dirx*std::max(1.0e-16,fabs(pgradx));
      double diry=1.0;
      if (pgradx < 0.0) diry=-1.0;
      pgrady = diry*std::max(1.0e-16,fabs(pgrady));
      double pgrad = sqrt(pgradx*pgradx+pgrady*pgrady);
      double xgrad = fabs(celldx(j)*pgrad/pgradx);
      double ygrad = fabs(celldy(k)*pgrad/pgrady);
      double grad  = std::min(xgrad,ygrad);
      double grad2 = grad*grad;

      viscosity(j,k)=2.0*density0(j,k)*grad2*limiter*limiter;
    }
  });
}


//  @brief Driver for the viscosity kernels
//  @author Wayne Gaudin
//  @details Selects the user specified kernel to caluclate the artificial 
//  viscosity.
void viscosity(global_variables& globals) {

  for (int tile = 0; tile < globals.tiles_per_chunk; ++tile) {

    viscosity_kernel(
      globals.chunk.tiles[tile].t_xmin,
      globals.chunk.tiles[tile].t_xmax,
      globals.chunk.tiles[tile].t_ymin,
      globals.chunk.tiles[tile].t_ymax,
      globals.chunk.tiles[tile].field.celldx,
      globals.chunk.tiles[tile].field.celldy,
      globals.chunk.tiles[tile].field.density0,
      globals.chunk.tiles[tile].field.pressure,
      globals.chunk.tiles[tile].field.viscosity,
      globals.chunk.tiles[tile].field.xvel0,
      globals.chunk.tiles[tile].field.yvel0);
  }
}
