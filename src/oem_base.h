#ifndef OEM_BASE_H
#define OEM_BASE_H

#include <RcppEigen.h>
#include "utils.h"
#include <limits>
#include <cmath>


template<typename VecTypeBeta>
class oemBase
{
protected:
    
    const int nvars;                  // dimension of beta
    int nobs;                         // number of rows
    const int ngroups;                // number of groups for group lasso
    
    bool intercept;                   //
    bool standardize;                 //
    
    double meanY;
    double scaleY;
    
    VectorXd u;                       // u vector
    
    VecTypeBeta beta;                 // parameters to be optimized
    VecTypeBeta beta_prev;            // parameters from previous iteration
    VecTypeBeta beta_prev_irls;       // parameters from previous irls iteration
    
    Eigen::RowVectorXd colmeans;      // column means of X
    Eigen::RowVectorXd colstd;        // column std devs of X
    
    double tol;                       // tolerance for convergence
    
    // Multiplier on the ridge part of the ".net" penalties. For Gaussian
    // models the drivers set it to 1 / s_y, where s_y is the standard
    // deviation of y (root mean square without an intercept), which is the
    // convention of glmnet. It is 1 for other families.
    double ridge_scale = 1.0;
    
    int accel_method;                 // 0 = none, 1 = FISTA, 2 = SQUAREM (S1), 3 = Anderson
    int accel_mem;                    // memory for Anderson acceleration
    int n_u_evals;                    // evaluations of next_u() in the last solve
    VectorXd lin_term;                // linear term c of the quadratic surrogate
    double obj_d;                     // d used by the surrogate objective
    
    // Whether next_u() is affine in beta_prev within one call of inner_solve().
    // Subclasses for which this fails must return false.
    virtual bool u_affine_in_beta() { return true; }
    
    // Whether the current penalty is convex. Nonconvex penalties get a
    // monotone safeguard in every accelerated method. The default is
    // conservative: a subclass that does not override this and
    // penalty_value() runs the plain iteration.
    virtual bool penalty_is_convex() { return false; }
    
    // Penalty value P(b) for the current penalty and lambda. NaN means it is
    // not available, in which case methods that need the objective fall back
    // to the plain iteration.
    virtual double penalty_value(const VecTypeBeta &b) 
    { 
        return std::numeric_limits<double>::quiet_NaN(); 
    }
    
    // Fills c with the linear term of the surrogate if it is cheaply
    // available. Otherwise c = u(0) is computed at the cost of one next_u().
    virtual bool linear_term(VectorXd &c) { return false; }
    
    virtual void next_beta(VecTypeBeta &res) = 0;
    
    virtual void next_u(VectorXd &res) = 0;
    
    virtual bool converged()
    {
        return (stopRule(beta, beta_prev, tol));
    }
    
    
    void print_row(int iter)
    {
        const char sep = ' ';
        
        Rcpp::Rcout << std::left << std::setw(7)  << std::setfill(sep) << iter;
        Rcpp::Rcout << std::endl;
    }
    
    
public:
    oemBase(int n_, 
            int p_,
            int ngroups_,
            bool intercept_,
            bool standardize_,
            double tol_ = 1e-6) :
    nvars(p_), 
    nobs(n_),
    ngroups(ngroups_),
    intercept(intercept_),
    standardize(standardize_),
    u(p_),               // allocate space but do not set values
    beta(p_),            // allocate space but do not set values
    beta_prev(p_),       // allocate space but do not set values
    beta_prev_irls(p_),
    colmeans(p_),
    colstd(p_),
    tol(tol_),
    accel_method(0),
    accel_mem(5),
    n_u_evals(0),
    obj_d(0.0)
    {}
    

    // Scalar MCP and SCAD penalties with level lam and parameter gamma,
    // matching the thresholding rules used by the solvers.
    static double mcp_val(double t, double lam, double gamma)
    {
        t = std::abs(t);
        if (t <= gamma * lam) return lam * t - t * t / (2.0 * gamma);
        return 0.5 * gamma * lam * lam;
    }
    static double scad_val(double t, double lam, double gamma)
    {
        t = std::abs(t);
        if (t <= lam) return lam * t;
        if (t <= gamma * lam) return (2.0 * gamma * lam * t - t * t - lam * lam) / (2.0 * (gamma - 1.0));
        return 0.5 * lam * lam * (gamma + 1.0);
    }
    
    // Penalty value shared by the Gaussian solvers. The ridge part of the
    // ".net" penalties applies to all coefficients, as in the thresholding
    // rules (it enters through the denominator d + (1 - alpha) * lambda).
    // Group 0 is unpenalized. Returns NaN for an unknown penalty.
    static double penalty_value_generic(const VectorXd &b, const std::string &pen,
                                        double lambda, double alpha, double gamma, double tau,
                                        const VectorXd &pf, const VectorXd &gw,
                                        const std::vector<std::vector<int> > &grp_idx,
                                        const VectorXi &unique_grps, double ridge_scale)
    {
        const int p = b.size();
        double val = 0.0;
        bool net = pen.find(".net") != std::string::npos;
        bool grp = pen.find("grp") != std::string::npos;
        double lam = net ? lambda * alpha : lambda;
        // Ridge part, as applied by the thresholding steps: multiplied by the
        // penalty factors for elementwise penalties, and omitted for group 0
        // (unpenalized coefficients) for group penalties.
        double ridge = 0.0;
        if (net)
        {
            double ss = 0.0;
            if (grp)
            {
                for (std::size_t g = 0; g < grp_idx.size(); ++g)
                {
                    if (unique_grps(g) == 0) continue;
                    for (std::size_t v = 0; v < grp_idx[g].size(); ++v) ss += b(grp_idx[g][v]) * b(grp_idx[g][v]);
                }
            } else
            {
                for (int i = 0; i < p; ++i) ss += pf(i) * b(i) * b(i);
            }
            ridge = 0.5 * (1.0 - alpha) * lambda * ridge_scale * ss;
        }
        
        if (pen == "ols") return 0.0;
        if (pen == "elastic.net") 
        {
            for (int i = 0; i < p; ++i) val += pf(i) * lam * std::abs(b(i));
            return val + ridge;
        }
        if (pen == "lasso" || pen == "mcp" || pen == "scad" || pen == "mcp.net" || pen == "scad.net")
        {
            for (int i = 0; i < p; ++i)
            {
                double l = pf(i) * lam;
                if (pen == "lasso") val += l * std::abs(b(i));
                else if (pen.compare(0, 3, "mcp") == 0) val += (l > 0 ? mcp_val(b(i), l, gamma) : 0.0);
                else val += (l > 0 ? scad_val(b(i), l, gamma) : 0.0);
            }
            return val + ridge;
        }
        if (pen.find("grp") != std::string::npos)
        {
            double lg = lam;
            if (pen == "sparse.grp.lasso")
            {
                lg = (1.0 - tau) * lambda;
                for (int i = 0; i < p; ++i) val += pf(i) * tau * lambda * std::abs(b(i));
            }
            for (std::size_t g = 0; g < grp_idx.size(); ++g)
            {
                if (unique_grps(g) == 0) continue;
                double nrm = 0.0;
                for (std::size_t v = 0; v < grp_idx[g].size(); ++v) nrm += b(grp_idx[g][v]) * b(grp_idx[g][v]);
                nrm = std::sqrt(nrm);
                double l = gw(g) * lg;
                if (pen == "grp.mcp" || pen == "grp.mcp.net") val += mcp_val(nrm, l, gamma);
                else if (pen == "grp.scad" || pen == "grp.scad.net") val += scad_val(nrm, l, gamma);
                else val += l * nrm;
            }
            return val + ridge;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    void set_accel(int method_, int mem_ = 5)
    {
        accel_method = method_;
        accel_mem    = mem_;
    }
    
    virtual ~oemBase() {}
    
    void update_u()
    {
        //VecTypeBeta newbeta(nvars);
        next_u(u);
        //beta.swap(newbeta);
    }
    
    void update_beta()
    {
        //VecTypeBeta newbeta(nvars);
        next_beta(beta);
        //beta.swap(newbeta);
    }
    
    // Solve the problem for the current lambda. Returns the number of
    // evaluations of next_u(), which equals the number of OEM iterations
    // for the plain method and is the dominant per-iteration cost for all
    // methods.
    virtual int solve(int maxit)
    {
        return inner_solve(maxit);
    }
    
protected:
    
    // Inner OEM iteration for one fixed surrogate problem. Subclasses with an
    // outer IRLS/MM loop can call this in place of their own inline loop.
    // The accelerated methods assume that, within a single call, next_u() is
    // an affine function of beta_prev, that is
    //     u(b) = d * b - H * b + c
    // with H symmetric and d fixed. This holds for every Gaussian solver and
    // for the inner weighted least squares problem of each IRLS step, because
    // the weights and working response do not change inside one call.
    int inner_solve(int maxit)
    {
        n_u_evals = 0;
        
        if (accel_method == 0 || !u_affine_in_beta())
        {
            return plain_solve(maxit);
        }
        
        // the objective is needed by SQUAREM, Anderson, and by FISTA
        // for nonconvex penalties
        bool need_obj = (accel_method != 1) || !penalty_is_convex();
        if (need_obj && !setup_objective())
        {
            return plain_solve(maxit);
        }
        
        if (accel_method == 1)
        {
            return fista_solve(maxit);
        } else if (accel_method == 2)
        {
            return squarem_solve(maxit);
        } else if (accel_method == 3)
        {
            return anderson_solve(maxit);
        }
        return plain_solve(maxit);
    }
    
    // Original OEM loop.
    int plain_solve(int maxit)
    {
        int i;
        
        for(i = 0; i < maxit; ++i)
        {
            
            beta_prev = beta;
            
            update_u();
            
            update_beta();
            
            if(converged())
                break;
            
        }
        
        n_u_evals = std::min(i + 1, maxit);
        return i + 1;
    }
    
    // u at an arbitrary point b. Counts as one iteration's worth of work.
    void eval_u(const VecTypeBeta &b, VectorXd &ub)
    {
        beta_prev = b;
        next_u(ub);
        ++n_u_evals;
    }
    
    // Thresholding (proximal) step applied to a given u. The result is
    // always an output of next_beta(), so it is exactly sparse.
    void eval_prox(const VectorXd &ub, VecTypeBeta &b)
    {
        u = ub;
        next_beta(beta);
        b = beta;
    }
    
    // Prepare the linear term c of the quadratic surrogate. Returns false
    // when the objective cannot be evaluated.
    // Work vectors are sized from beta rather than nvars, because some
    // solvers (for example oemSparse) store the intercept in beta but do not
    // count it in nvars.
    bool setup_objective()
    {
        obj_d = get_d();
        if (!(obj_d > 0.0)) return false;
        
        VecTypeBeta bz(beta.size());
        bz.setZero();
        if (std::isnan(penalty_value(bz))) return false;
        
        if (!linear_term(lin_term))
        {
            // c = u(0). Costs one extra evaluation of next_u().
            lin_term.resize(beta.size());
            eval_u(bz, lin_term);
        }
        return true;
    }
    
    // Penalized surrogate objective at b, given ub = u(b):
    //   0.5 * b' H b - c' b + P(b)  =  0.5 * b'(d b - ub) - 0.5 * c' b + P(b),
    // since H b = d b - ub + c. Costs O(p).
    double eval_obj(const VecTypeBeta &b, const VectorXd &ub)
    {
        return 0.5 * b.dot(obj_d * b - ub) - 0.5 * lin_term.dot(b) + penalty_value(b);
    }
    
    // tolerance for the monotone check, relative to the objective scale
    static bool obj_increased(double fnew, double fold)
    {
        return fnew > fold + 1e-12 * std::max(1.0, std::abs(fold));
    }
    
    static bool same_support(const VecTypeBeta &a, const VecTypeBeta &b)
    {
        for (int i = 0; i < a.size(); ++i)
        {
            if ((a(i) == 0.0) != (b(i) == 0.0)) return false;
        }
        return true;
    }
    
    // Store the final iterate.
    void finish(const VecTypeBeta &bprev, const VecTypeBeta &bfinal)
    {
        beta_prev = bprev;
        beta = bfinal;
    }
    
    // FISTA with adaptive restart (O'Donoghue and Candes, 2015).
    // x is the sequence of thresholded iterates, y the extrapolated points.
    // Because u is affine in b and the extrapolation weights sum to one,
    // u(y) = (1 + m) u(x_new) - m u(x), so each iteration costs one next_u().
    // Convex penalties use the gradient restart test
    //   (y - x_new)'(x_new - x) > 0.
    // Nonconvex penalties additionally require the objective not to increase;
    // otherwise the plain OEM step from x is taken and momentum is reset.
    int fista_solve(int maxit)
    {
        const bool monotone = !penalty_is_convex();
        
        VecTypeBeta x = beta, xn(beta.size()), xT(beta.size()), y(beta.size());
        VectorXd ux(beta.size()), uxn(beta.size()), uy(beta.size());
        
        eval_u(x, ux);
        double Fx = monotone ? eval_obj(x, ux) : 0.0;
        
        // plain convergence check at the starting point
        eval_prox(ux, xn);
        if (stopRule(xn, x, tol) || n_u_evals >= maxit)
        {
            finish(x, xn);
            return n_u_evals;
        }
        
        double t = 1.0;
        y = x; 
        uy = ux;
        bool first = true;
        
        while (n_u_evals < maxit)
        {
            if (!first) eval_prox(uy, xn);
            first = false;
            
            eval_u(xn, uxn);
            
            bool restart = false;
            if (monotone)
            {
                double Fxn = eval_obj(xn, uxn);
                if (obj_increased(Fxn, Fx))
                {
                    // reject the accelerated point, take the plain step from x
                    eval_prox(ux, xn);
                    eval_u(xn, uxn);
                    Fxn = eval_obj(xn, uxn);
                    restart = true;
                }
                Fx = Fxn;
            }
            
            if (!restart && (y - xn).dot(xn - x) > 0.0)
            {
                restart = true;
            }
            
            // same stopping rule as the plain method, applied at xn
            eval_prox(uxn, xT);
            if (stopRule(xT, xn, tol))
            {
                finish(xn, xT);
                return n_u_evals;
            }
            
            double m;
            if (restart)
            {
                t = 1.0;
                m = 0.0;
            } else
            {
                double tn = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
                m = (t - 1.0) / tn;
                t = tn;
            }
            
            y  = xn + m * (xn - x);
            uy = uxn + m * (uxn - ux);
            x.swap(xn);
            ux.swap(uxn);
        }
        
        // maxit reached: return the last thresholded iterate
        finish(xn, x);
        return n_u_evals;
    }
    
    // SQUAREM (Varadhan and Roland, 2008) on the map T(b) = prox(u(b)),
    // with a step length bound that grows after successful steps and
    // a monotone safeguard: if the stabilized point does not have objective
    // at most that of the second plain iterate, the plain iterate is kept.
    // With r = x1 - x0 and v = x2 - 2 x1 + x0, the step length is that of
    // scheme S1, a = r'v / v'v, clamped to [-alpha_max, -1]. In benchmarks on
    // OEM paths, S1 needed fewer iterations than S2 and than S3 (the default
    // of the SQUAREM package) for convex penalties.
    // Each cycle costs three evaluations of next_u().
    int squarem_solve(int maxit)
    {
        VecTypeBeta x0 = beta, x1(beta.size()), x2(beta.size()), xp(beta.size()), xn(beta.size());
        VectorXd u0(beta.size()), u1(beta.size()), u2(beta.size()), up(beta.size()), un(beta.size());
        
        eval_u(x0, u0);
        double alpha_max = 1.0;
        
        while (true)
        {
            eval_prox(u0, x1);
            if (stopRule(x1, x0, tol) || n_u_evals >= maxit)
            {
                finish(x0, x1);
                return n_u_evals;
            }
            eval_u(x1, u1);
            eval_prox(u1, x2);
            if (stopRule(x2, x1, tol) || n_u_evals >= maxit)
            {
                finish(x1, x2);
                return n_u_evals;
            }
            eval_u(x2, u2);
            
            VectorXd r = x1 - x0;
            VectorXd v = x2 - 2.0 * x1 + x0;
            double rn = r.norm(), vn = v.norm();
            
            if (vn > 0.0 && n_u_evals < maxit)
            {
                double a = r.dot(v) / (vn * vn);
                if (!std::isfinite(a) || a > -1.0) a = -1.0;
                if (a < -alpha_max) a = -alpha_max;
                
                // xp = x0 - 2 a r + a^2 v; weights on (x0, x1, x2) sum to one
                double w0 = 1.0 + 2.0 * a + a * a;
                double w1 = -2.0 * a - 2.0 * a * a;
                double w2 = a * a;
                up = w0 * u0 + w1 * u1 + w2 * u2;
                
                // stabilization step
                eval_prox(up, xn);
                eval_u(xn, un);
                
                double F2 = eval_obj(x2, u2);
                double Fn = eval_obj(xn, un);
                if (!obj_increased(Fn, F2))
                {
                    if (a <= -alpha_max) alpha_max *= 4.0;
                    x0.swap(xn);
                    u0.swap(un);
                    continue;
                }
                alpha_max = std::max(1.0, alpha_max / 4.0);
            }
            
            x0.swap(x2);
            u0.swap(u2);
            if (n_u_evals >= maxit)
            {
                eval_prox(u0, x1);
                finish(x0, x1);
                return n_u_evals;
            }
        }
    }
    
    // Type II Anderson acceleration with memory accel_mem, applied in u space
    // to S(u) = u(prox(u)). The plain OEM iteration is u_{k+1} = S(u_k), and the
    // iterates b_k = prox(u_k) are always thresholded points. Safeguards:
    // an accelerated u_{k+1} is accepted only if F(prox(u_{k+1})) <= F(b_k);
    // otherwise the plain step u_{k+1} = S(u_k) is taken and the memory is
    // cleared. The memory is also cleared when the zero pattern of b changes,
    // since the map is only locally affine on a fixed support.
    int anderson_solve(int maxit)
    {
        const int mem = std::max(1, accel_mem);
        
        VecTypeBeta b = beta, bT(beta.size()), bn(beta.size());
        VectorXd s(beta.size()), sn(beta.size()), uk(beta.size()), un(beta.size());
        VectorXd r(beta.size()), rn(beta.size());
        
        MatrixXd dR(beta.size(), mem), dS(beta.size(), mem);
        int ncol = 0, head = 0;
        
        eval_u(b, s);
        double Fb = eval_obj(b, s);
        
        // plain step to obtain the first u iterate
        uk = s;
        eval_prox(uk, bT);
        if (stopRule(bT, b, tol) || n_u_evals >= maxit)
        {
            finish(b, bT);
            return n_u_evals;
        }
        b = bT;
        eval_u(b, s);
        Fb = eval_obj(b, s);
        r = s - uk;
        
        while (true)
        {
            // plain stopping rule at b: compare T(b) = prox(s) with b
            eval_prox(s, bT);
            if (stopRule(bT, b, tol) || n_u_evals >= maxit)
            {
                finish(b, bT);
                return n_u_evals;
            }
            
            // candidate u_{k+1}
            bool accelerated = false;
            if (ncol > 0)
            {
                // least squares for the mixing coefficients, with a small
                // Tikhonov term because the columns of dR become nearly
                // collinear close to convergence
                MatrixXd RtR = dR.leftCols(ncol).transpose() * dR.leftCols(ncol);
                VectorXd Rtr = dR.leftCols(ncol).transpose() * r;
                double reg = 1e-10 * std::max(RtR.trace(), 1e-300);
                RtR.diagonal().array() += reg;
                VectorXd g = RtR.ldlt().solve(Rtr);
                if (g.allFinite() && g.norm() < 1e4)
                {
                    un = s - dS.leftCols(ncol) * g;
                    accelerated = true;
                } else 
                {
                    ncol = 0; head = 0;
                }
            }
            if (!accelerated) un = s;
            
            if (accelerated)
            {
                eval_prox(un, bn);
                eval_u(bn, sn);
                double Fn = eval_obj(bn, sn);
                if (obj_increased(Fn, Fb))
                {
                    // reject: plain step, clear memory
                    accelerated = false;
                    ncol = 0; head = 0;
                    un = s;
                    if (n_u_evals >= maxit)
                    {
                        finish(b, bT);
                        return n_u_evals;
                    }
                } else 
                {
                                        if (!same_support(bn, b)) { ncol = 0; head = 0; }
                    Fb = Fn;
                }
            }
            if (!accelerated)
            {
                bn = bT; // prox(s)
                eval_u(bn, sn);
                double Fn = eval_obj(bn, sn);
                if (!same_support(bn, b)) { ncol = 0; head = 0; }
                Fb = Fn;
            }
            
            rn = sn - un;
            // store differences in a circular buffer
            dR.col(head) = rn - r;
            dS.col(head) = sn - s;
            head = (head + 1) % mem;
            if (ncol < mem) ++ncol;
            
            b.swap(bn);
            s.swap(sn);
            r.swap(rn);
            uk.swap(un);
        }
    }
    
public:
    
    virtual void init_oem() {}
    
    virtual void init_xtx(bool add_int_) {}
    virtual void update_xtx(int fold_) {}
    virtual double compute_lambda_zero() { return 0; }
    void set_ridge_scale(double r) { ridge_scale = (r > 0.0 && std::isfinite(r)) ? r : 1.0; }

    // Smallest lambda at which the current fit is a fixed point of the
    // thresholding step, so that no coefficient that is zero in the current
    // fit becomes nonzero. When the current fit is the solution for a very
    // large lambda, every penalized coefficient is zero and the unpenalized
    // ones are fitted, so the result is the smallest lambda whose solution
    // has all penalized coefficients equal to zero. The search uses each
    // class's own thresholding step, so it accounts for penalty factors,
    // group weights, the elastic net mixing parameter, and the intercept.
    // Returns a negative value if no finite lambda keeps the fit fixed,
    // as for a pure ridge penalty.
    double lambda_max_at_current_fit()
    {
        const VecTypeBeta beta_null = beta;
        beta_prev = beta;
        update_u();

        const double scale = u.cwiseAbs().maxCoeff();
        const double zero_tol = 1e-12 * (1.0 + beta_null.cwiseAbs().maxCoeff());

        // true if thresholding u at lam leaves every zero coefficient at zero
        auto stays_null = [&](double lam) {
            init_warm(lam);
            beta = beta_null;
            update_beta();
            for (int j = 0; j < beta.size(); ++j)
            {
                if (beta_null(j) == 0.0 && std::abs(beta(j)) > zero_tol) return false;
            }
            return true;
        };

        double result = -1.0;
        if (!(scale > 0.0))
        {
            result = 0.0;
        } else
        {
            double hi = scale;
            int k = 0;
            while (!stays_null(hi) && k < 60)
            {
                hi *= 2.0;
                ++k;
            }
            if (stays_null(hi))
            {
                double lo = 0.0;
                for (int it = 0; it < 100 && (hi - lo) > 1e-10 * hi; ++it)
                {
                    double mid = 0.5 * (lo + hi);
                    if (stays_null(mid)) hi = mid;
                    else lo = mid;
                }
                // The boundary is exact only at a fully converged null fit.
                // A small relative margin keeps every penalized coefficient
                // at exactly zero in the first fit of the path when that fit
                // is converged only to tol, as with accelerated iterations.
                result = hi * (1.0 + 1e-6);
            }
        }
        beta = beta_null;
        beta_prev = beta_null;
        return result;
    }
    virtual VecTypeBeta get_beta() { return beta; }
    virtual double get_d() { return 0; }
    
    Eigen::RowVectorXd get_X_colmeans() {return colmeans;}
    Eigen::RowVectorXd get_X_colstd() {return colstd;}
    
    virtual double get_loss() { return 1e99; }
    
    virtual void init(double lambda_, std::string penalty_,
                      double alpha_, double gamma_, double tau_) {}
    virtual void init_warm(double lambda_) {}
};


// Largest lambda of the default sequence for one penalty: the smallest lambda
// at which every penalized coefficient is zero. The solver first fits the model
// in which only the unpenalized coefficients (and the intercept) are free, by
// fitting at a very large lambda with the L1 form of the penalty family. Lasso,
// MCP, and SCAD share the zero region |u_j| <= lambda * pf_j, and the group
// penalties share ||u_g|| <= lambda * w_g, so the L1 form gives the same value.
// The ".net" penalties apply alpha * lambda to the L1 part, so their value is
// divided by alpha. Returns a negative value when there is no such lambda (for
// "ols", or alpha = 0), in which case the caller keeps its previous sequence.
template<typename SolverType>
inline double oem_lambda_max(SolverType *solver, const std::string &penalty,
                             double alpha, double gamma, double tau, int maxit)
{
    if (penalty == "ols") return -1.0;

    std::string base_penalty("lasso");
    if (penalty == "sparse.grp.lasso")
    {
        base_penalty = penalty;
    } else if (penalty.compare(0, 4, "grp.") == 0)
    {
        base_penalty = "grp.lasso";
    }

    solver->init(1e100, base_penalty, alpha, gamma, tau);
    solver->solve(maxit);
    double lmax = solver->lambda_max_at_current_fit();

    if (lmax > 0.0 && penalty.find(".net") != std::string::npos)
    {
        lmax = (alpha > 0.0) ? lmax / alpha : -1.0;
    }
    return lmax;
}

// Log-spaced sequence from lmax down to lmin_ratio * lmax.
inline Eigen::VectorXd oem_lambda_sequence(double lmax, double lmin_ratio, int nlambda)
{
    Eigen::VectorXd out(nlambda);
    out.setLinSpaced(nlambda, std::log(lmax), std::log(lmin_ratio * lmax));
    return out.array().exp().matrix();
}


#endif // OEM_BASE_H