#EPIONCHO-IBM 
# this script tests different ABRS (with different k_E parameters) to get endemic eqs
# 05-12-22
#Jonathan Hamley, Matt Dixon

#individual-based at the level of humans (but not parasites), age-structued for adult O.volvulus and microfilariae
#parasite life stages in black flies are deterministic and population-based
#all functions are called in ep.equi.sim which gives the final output


#function to calculate which individuals will be treated
#all.dt is a matrix containing (among other things) information on compliance to treatment (whether or not an individual can be treated)
#and the age of each individual 
#covrg is the total coverage of the population

#iter <- as.numeric(Sys.getenv("PBS_ARRAY_INDEX")) # iter will be a number from total runs across all parameter combos etc (e.g., 10 abrs x 100 = 1000 iters; 10 abrs x 5 k_Es = 5000 iters)

#set.seed(iter + (iter*3758))


os.cov <- function(all.dt, pncomp, covrg, N) 
  
{                   
  pop.ages <- all.dt[,2] #age of each individual in population 
  
  iny <- which(pop.ages < 5 | all.dt[,1] == 1) 
  
  nc.age <- length(iny) / length(pop.ages)
  
  covrg <- covrg / (1 - nc.age) #probability a complying individual will be treated
  
  out.cov <- rep(covrg, length(pop.ages))
  
  out.cov[iny] <- 0 #non compliers get probability 0
  
  f.cov <- rep(0, N)
  
  r.nums <- runif(N, 0, 1)
  
  inds.c <- which(r.nums < out.cov)
  
  f.cov[inds.c] <- 1
  
  return(f.cov)
  
}



#function to rotate matrix, used in mf function 
rotate <- function(x) t(apply(x, 2, rev))

#function calculates change in the number of microfilariae (mf) (offspring of adult worms) in each human using RK4 method
#this is called in ep.equi.sim for each mf age class
#num.comps is the number of age classes in adult worms (fecundity rate is a function of adult worm age)
#num.mf.comps is the number of microfilarial age classes
#mf.cpt is the age class under consideration 
#mf mortality rate is a function of age, mu.rates.mf contains the mortality rate for each mf age class
#mf.move.rate determines aging (moving to the next age class)
#up and kap determine extent of ivermectin (drug given to people to treat onchocerciasis) induced mortality
#treat.vec contains how long since a person was treated, mortality rate due to ivermectin decays with time since treatment
change.micro <- function(dat, num.comps, mf.cpt, num.mf.comps, ws, DT, time.each.comp, mu.rates.mf, fec.rates, mf.move.rate,
                         up, kap, iteration, treat.vec, give.treat, treat.start)
  
{
  N <- length(dat[,1])
  #indexes for fertile worms (to use in production of mf)
  fert.worms.start <-  ws + num.comps*2 
  fert.worms.end <-  (ws-1) + num.comps*3
  
  #indexes to check if there are males (males start is just 'ws')
  #there must be >= 1 male worm for females to produce microfilariae 
  mal.worms.end <- (ws-1) + num.comps
  mf.mu <- rep(mu.rates.mf[mf.cpt], N)
  fert.worms <- dat[, fert.worms.start:fert.worms.end] #number of fertile females worms
  
  #increases microfilarial mortality if treatment has started
  if(give.treat == 1 & iteration >= treat.start)
  {
    tao <- ((iteration-1)*DT) - treat.vec #tao is zero if treatment has been given at this timestep
    
    mu.mf.prime <- ((tao + up) ^ (-kap)) #additional mortality due to ivermectin treatment
    
    mu.mf.prime[which(is.na(mu.mf.prime) == TRUE)] <- 0
    
    mf.mu <- mf.mu + mu.mf.prime
    
  }
  
  if(mf.cpt == 1) #if the first age class of microfilariae
  {
    mp <- rep(0, N)
    
    inds.fec <- which(rowSums(dat[, ws : mal.worms.end]) > 0); mp[inds.fec] <- 1     #need to check there is more than one male
    
    k1 <- derivmf.one(fert.worms = fert.worms, mf.in = dat[, 6 + mf.cpt], ep.in = fec.rates, mf.mort = mf.mu, mf.move = mf.move.rate, mp = mp, k.in = 0)  #fert worms and epin are vectors
    k2 <- derivmf.one(fert.worms = fert.worms, mf.in = dat[, 6 + mf.cpt], ep.in = fec.rates, mf.mort = mf.mu, mf.move = mf.move.rate, mp = mp, k.in = DT*k1/2) 
    k3 <- derivmf.one(fert.worms = fert.worms, mf.in = dat[, 6 + mf.cpt], ep.in = fec.rates, mf.mort = mf.mu, mf.move = mf.move.rate, mp = mp, k.in = DT*k2/2)  
    k4 <- derivmf.one(fert.worms = fert.worms, mf.in = dat[, 6 + mf.cpt], ep.in = fec.rates, mf.mort = mf.mu, mf.move = mf.move.rate, mp = mp, k.in = DT*k3) 
    
    out <- dat[, 6 + mf.cpt] + DT/6*(k1+2*k2+2*k3+k4)
    
  }
  
  if(mf.cpt > 1) #if age class of microfilariae is >1
  {
    k1 <- derivmf.rest(mf.in = dat[, 6 + mf.cpt], mf.mort = mf.mu, mf.move = mf.move.rate, mf.comp.minus.one = dat[, 6 + mf.cpt - 1], k.in = 0) 
    k2 <- derivmf.rest(mf.in = dat[, 6 + mf.cpt], mf.mort = mf.mu, mf.move = mf.move.rate, mf.comp.minus.one = dat[, 6 + mf.cpt - 1], k.in = DT*k1/2) 
    k3 <- derivmf.rest(mf.in = dat[, 6 + mf.cpt], mf.mort = mf.mu, mf.move = mf.move.rate, mf.comp.minus.one = dat[, 6 + mf.cpt - 1], k.in = DT*k2/2) 
    k4 <- derivmf.rest(mf.in = dat[, 6 + mf.cpt], mf.mort = mf.mu, mf.move = mf.move.rate, mf.comp.minus.one = dat[, 6 + mf.cpt - 1], k.in = DT*k3) 
    
    out <- dat[, 6 + mf.cpt] + DT/6*(k1+2*k2+2*k3+k4)
    
  }  
  
  
  return(out)
}

#function called during RK4 for first age class of microfilariae 
#ep.in is fecundity of female worms
derivmf.one <- function(fert.worms, mf.in, ep.in, mf.mort, mf.move, mp, k.in)  #fert worms and epin are vectors
{
  
  new.in <- (rotate(fert.worms)*ep.in) #need to rotate matrix to each column is multiplied by respective fecundity rate, not each row
  new.in <- rotate(rotate(rotate(new.in)))
  new.in <- rowSums(new.in)
  
  mort.temp <- mf.mort*(mf.in + k.in)
  move.temp <- mf.move * (mf.in + k.in) 
  
  mort.temp[which(mort.temp < 0)] <- 0
  move.temp [which(move.temp < 0)] <- 0

  if(length(which(mf.mort*(mf.in + k.in) < 0)) >0) {print('MF NEGATIVE1')}
  if(length(which(mf.move * (mf.in + k.in) < 0)) >0) {print('MF NEGATIVE2')}

  out <- mp * new.in - mort.temp - move.temp
  
  # if(length(which(out < 0)) >0) {print('WARNING MF out')}
  # out[which(out < 0)] <- 0
  return(out)
}

#function called during RK4 for age classes of microfilariae > 1
derivmf.rest <- function(mf.in, mf.mort, mf.move, mf.comp.minus.one, k.in) 
{

  #out <- mf.comp.minus.one*mf.move - mf.mort*(mf.in + k.in) - mf.move * (mf.in + k.in)
  
  move.last <- mf.comp.minus.one*mf.move
  mort.temp <- mf.mort * (mf.in + k.in)
  move.temp <- mf.move * (mf.in + k.in)
  move.last[which(move.last < 0)] <- 0
  mort.temp[which(mort.temp < 0)] <- 0
  move.temp [which(move.temp < 0)] <- 0
  
  if(length(which(mf.mort*(mf.in + k.in) < 0)) >0) {print('WARNING MF NEGATIVE3')}
  if(length(which(mf.move * (mf.in + k.in) < 0)) >0) {print('WARNING MF NEGATIVE4')}
  
  
  out <- move.last - mort.temp - move.temp

  # if(length(which(out < 0)) >0) {print('WARNING MF out')}
  # 
  # out[which(out < 0)] <- 0

  return(out)
}


#proportion of L3 larvae (final life stage in the fly population) developing into adult worms in humans
#expos is the total exposure for an individual
#delta.hz, delta.hinf, c.h control the density dependent establishment of parasites
# matt: establishment of L? larvae in human hosts is negatively density (transmission intensity: ATP) dependent i.e. per capita rate of worm establishment decreasing with incr. ATP
# matt: controlled by function PI_H(i) (ATP(time - time delay between L3 entering/establishment), omega_t = individuals total exposure taking into account delay)

delta.h <- function(delta.hz, delta.hinf, c.h, L3, m , beta, expos)
  
{
  out <- (delta.hz + delta.hinf * c.h * m * beta *  L3 * expos) / (1 + c.h * m * beta * L3 * expos) 
  return(out)
}

#proportion of mf per mg developing into infective larvae within the vector (matt: Table F in Supp)

delta.v <- function(delta.vo, c.v, mf, expos)
  
{
  out <- delta.vo / (1 + c.v * mf *expos)
  
  return(out)
}

#L1, L2, L3 (parasite life stages) dynamics in the fly population
#assumed to be at equilibrium
#delay of 4 days for parasites moving from L1 to L2

calc.L1 <- function(beta, mf, mf.delay.in, expos, delta.vo, c.v, nuone, mu.v, a.v, expos.delay)
  
{
  delta.vv <- delta.v(delta.vo, c.v, mf, expos)#density dependent establishment
  
  out <- (delta.vv * beta * expos *  mf)  / ((mu.v + a.v * mf*expos) + (nuone * exp (-(4/366) * (mu.v + (a.v * mf.delay.in*expos.delay)))))
  
  return(out)
}

calc.L2 <- function(nuone, L1.in, mu.v, nutwo, mf, a.v, expos) 
  
{
  out <- (L1.in * (nuone * exp (-(4/366) * (mu.v + (a.v * mf * expos))))) / (mu.v + nutwo)
  
  return(out)
}

calc.L3 <- function(nutwo, L2.in, a.H, g, mu.v, sigma.L0)
  
{
  out <- (nutwo * L2.in) / ((a.H / g) + mu.v + sigma.L0) 
  
  return(out)
}


#rate of acquisition of new infections in humans
#depends on mean number of L3 larvae in the fly population 

Wplus1.rate <- function(delta.hz, delta.hinf, c.h, L3, m , beta, expos, DT)
  
{
  dh <- delta.h(delta.hz, delta.hinf, c.h, L3, m , beta, expos) # matt: density-dep (L3 establishing in human host)
  
  out <- DT * m * beta * dh * expos * L3 # matt: individual rate of acquisition of male / female adult worms - distrecte-time stochastic process with this probablility (pg.7 SI) ?
  
  return(out)
}



#people are tested for the presence of mf using a skin snip, we assume mf are overdispersed in the skin
#function calculates number of mf in skin snip for all people
#ss.wt is the weight of the skin snip
mf.per.skin.snip <- function(ss.wt, num.ss, slope.kmf, int.kMf, data, nfw.start, fw.end,  ###check vectorization 
                             mf.start, mf.end, pop.size)
  
{
  
  all.mfobs <- c()
  
  kmf <- slope.kmf * (rowSums(data[,nfw.start:fw.end])) + int.kMf #rowSums(da... sums up adult worms for all individuals giving a vector of kmfs
  # kmf <- 15
  
  mfobs <- rnbinom(pop.size, size = kmf, mu = ss.wt * (rowSums(data[,mf.start:mf.end])))
  
  nans <- which(mfobs == 'NaN'); mfobs[nans] <- 0
  
  if(num.ss > 1)
    
  {
    
    tot.ss.mf <- matrix(, nrow = length(data[,1]), ncol = num.ss) # error?
    tot.ss.mf[,1] <- mfobs
    
    for(j in 2 : (num.ss)) #could be vectorized
      
    {
      
      temp <- rnbinom(pop.size, size = kmf, mu = ss.wt * (rowSums(data[,mf.start:mf.end])))
      
      nans <- which(temp == 'NaN'); temp[nans] <- 0
      
      tot.ss.mf[,j] <- temp
      
    }
    
    mfobs <- rowSums(tot.ss.mf)  
    
  } 
  
  mfobs <- mfobs / (ss.wt * num.ss) 
  # mfobs <- mfobs / (num.ss) 
  
  list(mean(mfobs), mfobs)
  
}


#calculates the change in the number of adult worms in one adult worm age class for all people
#if appropriate considers additional mortality and loss of fertility due to ivermectin treatment
change.worm.per.ind<- function(delta.hz, delta.hinf, c.h, L3, m , beta, compartment, total.dat, num.comps,
                               w.f.l.c, lambda.zero, omeg, expos, ws, DT, mort.rates, time.each.comp, new.worms.m, new.worms.nf.fo,
                               lam.m, phi, treat.stop, iteration, treat.int, treat.prob, cum.infer, treat.vec, give.treat, treat.start, N, onchosim.cov, times.of.treat)
  
{
  N <- length(treat.vec)
  
  lambda.zero.in <- rep(lambda.zero * DT, N) #loss of fertility
  omeg <- rep(omeg * DT, N) #becoming fertile
  
  #male worms
  
  cl <- (ws-1) + compartment #calculate which column to use depending on sex, type (fertile or infertile) and compartment
  
  cur.Wm <- total.dat[, cl] #take current number of worms from matrix
  
  worm.dead.males <- rbinom(N, cur.Wm, rep(mort.rates[compartment], N))
  worm.loss.males <- rbinom(N, (cur.Wm - worm.dead.males), rep((DT / time.each.comp), N))
  
  
  if(compartment == 1)
    
  {
    male.tot.worms <- cur.Wm + new.worms.m - worm.loss.males - worm.dead.males
  }
  
  if(compartment > 1)
    
  {
    male.tot.worms <- cur.Wm + w.f.l.c[[2]] - worm.loss.males - worm.dead.males
  }
  
  #female worms
  
  clnf <- (ws - 1) + num.comps + compartment #column for infertile females, num.comps skips over males
  
  clf <- (ws - 1) + 2*num.comps + compartment #column for fertile females, 2*num.comps skips over males and infertile females
  
  cur.Wm.nf <- total.dat[, clnf] #take current worm number from matrix infertile females 
  
  cur.Wm.f <- total.dat[, clf] #take current worm number from matrix fertile females
  
  mort.fems <- rep(mort.rates[compartment], N)
  
  ######### 
  #treatment
  #########
  
  #approach assumes individuals which are moved from fertile to non fertile class due to treatment re enter fertile class at standard rate 
  
  if(give.treat == 1 & iteration >= treat.start) 
  {
    
    if((sum(times.of.treat == iteration) == 1) & iteration <= treat.stop) #is it a treatment time
      
    {
      #print('TREATMENT GIVEN') 
      
      inds.to.treat <- which(onchosim.cov == 1) #which individuals will received treatment
      
      treat.vec[inds.to.treat]  <-  (iteration-1) * DT #alter time since treatment 
      #cum.infer is the proportion of female worms made permanently infertile, killed for simplicity 
      if(iteration > treat.start) {mort.fems[inds.to.treat] <- mort.fems[inds.to.treat] + (cum.infer); print('cum.infer')} #alter mortality 
    }
    
    
    tao <- ((iteration-1)*DT) - treat.vec #vector of toas, some will be NA
    
    lam.m.temp <- rep(0, N); lam.m.temp[which(is.na(treat.vec) != TRUE)] <- lam.m #individuals which have been treated get additional infertility rate
    
    f.to.nf.rate <- DT * (lam.m.temp * exp(-phi * tao)) #account for time since treatment (fertility reduction decays with time since treatment)
    
    f.to.nf.rate[which(is.na(treat.vec) == TRUE)] <- 0 #these entries in f.to.nf.rate will be NA, lambda.zero.in cannot be NA
    
    lambda.zero.in <- lambda.zero.in + f.to.nf.rate #update 'standard' fertile to non fertile rate to account for treatment 
    
  }
  ############################################################
  
  #.fi = 'from inside': worms moving from a fertile or infertile compartment
  #.fo = 'from outside': completely new adult worms 
  
  
  worm.dead.nf <- rbinom(N, cur.Wm.nf, mort.fems) #movement to next compartment
  
  worm.dead.f <- rbinom(N, cur.Wm.f, mort.fems)
  
  worm.loss.age.nf <- rbinom(N, (cur.Wm.nf - worm.dead.nf), rep((DT / time.each.comp), N))
  
  worm.loss.age.f <- rbinom(N, (cur.Wm.f - worm.dead.f), rep((DT / time.each.comp), N))
  
  
  #calculate worms moving between fertile and non fertile, deaths and aging 
  
  #females from fertile to infertile
  
  new.worms.nf.fi <- rep(0, N)
  
  trans.fc <- which((cur.Wm.f - worm.dead.f - worm.loss.age.f) > 0)
  
  #individuals which still have fertile worms in an age compartment after death and aging
  if(length(trans.fc) > 0) 
  {
    new.worms.nf.fi[trans.fc] <- rbinom(length(trans.fc), (cur.Wm.f[trans.fc] - worm.dead.f[trans.fc] - worm.loss.age.f[trans.fc]), lambda.zero.in[trans.fc])
  }
  
  
  #females worms from infertile to fertile, this happens independent of males, but production of mf depends on males
  
  #individuals which still have non fertile worms in an age compartment after death and aging
  new.worms.f.fi <- rep(0, N)
  
  trans.fc <-  which((cur.Wm.nf - worm.dead.nf - worm.loss.age.nf) > 0)
  if(length(trans.fc) > 0)
  {
    new.worms.f.fi[trans.fc] <- rbinom(length(trans.fc), (cur.Wm.nf[trans.fc] - worm.dead.nf[trans.fc] - worm.loss.age.nf[trans.fc]), omeg[trans.fc])#females moving from infertile to fertile
  }
  
  
  
  if(compartment == 1) #if it's the first adult worm age compartment 
    
  {
    nf.out <- cur.Wm.nf + new.worms.nf.fo + new.worms.nf.fi - worm.loss.age.nf - new.worms.f.fi - worm.dead.nf#final number of infertile worms
    
    f.out <- cur.Wm.f + new.worms.f.fi - worm.loss.age.f - new.worms.nf.fi - worm.dead.f#final number of fertile worms
  }       
  
  if(compartment > 1) 
    
  {
    nf.out <- cur.Wm.nf + new.worms.nf.fi - worm.loss.age.nf - new.worms.f.fi + w.f.l.c[[5]] - worm.dead.nf#w.f.l.c = worms from previous compartment
    
    f.out <- cur.Wm.f + new.worms.f.fi - worm.loss.age.f - new.worms.nf.fi + w.f.l.c[[6]] - worm.dead.f
  }   
  
  
  list(male.tot.worms,
       worm.loss.males,
       nf.out,
       f.out,
       worm.loss.age.nf,
       worm.loss.age.f, treat.vec)  
}


#age-dependent mortality for adult worms and microfilariae
weibull.mortality <- function(DT, par1, par2, age.cats)
  
{
  out <- DT  * (par1 ^ par2) * par2 * (age.cats ^ (par2-1))
  return(out)
}

#mf prevalence in people based on a skin snip
prevalence.for.age <- function(age, ss.in, main.dat)
  
{
  inds <- which(main.dat[,2] >= age)
  
  out <- length(which(ss.in[[2]][inds] > 0)) / length(inds) 
  
  return(out)
}


#ep.equi.sim will run one repeat of the model, typically the mean of 500 repeats is required
#model must be run to equilibrium (100 years), before treatment can begin 
#treatment is with ivermectin
#if the mf prevalence is zero 50 years after the final treatment, we assume elimination has occured
#code is available which saves the equilibrium and receives it as an input 

#time.its = number of iterations, ABR = annual biting rate, treat.int = treatment interval, treat.prob = total population coverage
#give.treat takes 1 (MDA) or 0 (no MDA), pnc = proportion of population which never receive treatment, min.mont.age is the minimum age for giving a skin snip




ep.equi.sim <- function(time.its,
                        ABR,
                        DT,
                        treat.int,
                        treat.prob,
                        give.treat,
                        treat.start,
                        treat.stop,
                        pnc,
                        min.mont.age,
                        delta.hz.in, # these inputs are new (matt) for testing DD
                        delta.hinf.in,
                        c.h.in,
                        gam.dis.in,
                        all.mats.temp.in,
                        exposure.delay.in,
                        ex.vec.in,
                        l1.delay.in,
                        l.extras.in,
                        mf.delay.in)


{ 
  
  # ================ #
  # hard coded parms #
  
  # density dep pars (worm establishment in humans)
  #delta.hz <- 0.1864987
  delta.hz <- delta.hz.in
  #delta.hinf <- 0.002772749
  delta.hinf <- delta.hinf.in
  #c.h <-  0.004900419
  c.h <- c.h.in
  
  m = ABR * ((1/104) / 0.63) # matt: m = vector to host ratio (V/H) ?; ABR = beta * V/H, where V/H = ABR/beta or V/H = ABR/(h/g), or V/H = ABR * (g/h) which you see here (note, V/H is inferred from the ABR - KEY INPUT for adjusting endemicity level of EPIONCHO-IBM sims)
  beta = 0.63 / (1/104) # matt: beta = per blackfly biting rate on humans (h/g; where h is the human blood index, g is duration of the gonotrophic cycle)
  mu.v = 26
  int.mf = 0
  sigma.L0 = 52
  a.H = 0.8
  g = 0.0096
  a.v = 0.39
  real.max.age = 80 #no humans live longer than 80 years
  N = 440 #human population size
  mean.age = 50 #mean human age (before truncation)
  int.L3 = 0.03; int.L2 = 0.03; int.L1 = 0.03
  lambda.zero = 0.33 # (matt:) per-capita rate that female worms lose their fertility (W_FF) & return to non-fertile state (W_FN)
  omeg = 0.59 # (matt:) per-capita rate that female worms progress from non-fertile (W_FN) to fertile (W_FF)
  delta.vo = 0.0166 # matt : delta V0, density dependence when microfilarae tend to 0
  c.v = 0.0205 # matt: severity of constraining density-dep larval development (per dermal mf) : Table F Supp
  num.mf.comps = 21; num.comps.worm = 21 #number of age classes for mf and adult worms (matt: c_max ?) 
  time.each.comp.worms = 1; time.each.comp.mf = 0.125; mf.move.rate = 8.133333 #for aging in parasites (matt: time.each.comp.worms = q_W & time.each.comp.mf = q_M in supp table E )
  int.worms=1 #initial number of worms in each worm age compartment
  ss.wt = 2; num.ss = 2 #skin snip parameters (matt weight and number)
  slope.kmf = 0.0478 # matt: parameter associated with decreasing degree of aggregation of skin mf with increasing no. of adult female worms (slope in linear model -> k_M = 0.0478 * W_F + 0.313)
  int.kMf = 0.313 # matt: parameter associated with decreasing degree of aggregation of skin mf with increasing no. of adult female worms (inital y value in linear model -> k_M = 0.0478 * W_F + 0.313)
  
  sex.rat = 0.5 #sex ratio (matt: inidividual assigned a sex randomly - equal probability psi_F = 0.5, psi_m = 0.5)
  
  nuone = 201.6189; nutwo = 207.7384 #movement of fly parasite life stages
  
  mu.w1 = 0.09953; mu.w2 = 6.00569 #parameters controlling age-dependent mortality in adult worms (matt: these are y_l = y_w and d_l = d_w in equation S6/S7 & Table E)
  mu.mf1 = 1.089; mu.mf2 = 1.428 #parameters controlling age-dependent mortality in mf (matt: these are y_l = y_m and d_l = d_m in equation S6/S7 & Table E)
  fec.w.1 = 70 ; fec.w.2 = 0.72 #parameters controlling age-dependent fecundity in adult worms (matt: fec.w.1 = F and fec.w.2 = G in Supp table E)
  l3.delay = 10; dt.days = DT*366 #delay in worms entering humans and joining the first adult worm age class (dt.days = DT.in*366)
  lam.m = 32.4; phi = 19.6 #effects of ivermectin (matt: embryostatic effect - lam.m is the max rate of treatment-induced sterility; phi is the rate of decay of this effect - Table G in Supp)
  cum.infer= 0.345 #permenent infertility in worms due to ivermectin 
  up = 0.0096; kap = 1.25 #effects of ivermectin (matt: parameters u (up) and k (kap) define the microfilaricidal effect curve, u = finite effect follwoed by decline (rebound) = k - table G in Supp)
  # gam.dis = 0.3 #individual level exposure heterogeneity (matt: shape par in gamma dist, K_E)
  gam.dis <- gam.dis.in # when specifying user input (K_E)
  E0 = 0; q = 0; m.exp = 1.08; f.exp = 0.9; age.exp.m = 0.007; age.exp.f = -0.023 #age-dependent exposure to fly bites age.exp.m or .f = alpha_m or alpha_f)
  
  # loop starts here ?
  
  list <- list()
  
    # 
    if(give.treat == 1) #calculate timesteps at which treatment is given
    {times.of.treat.in <- seq(treat.start, treat.stop - (treat.int / DT), treat.int / DT)}
    else {times.of.treat.in <- 0}

    # #columns to set to zero when an individual dies
    cols.to.zero <- seq(from = 1, to = (6 + num.mf.comps + 3*num.comps.worm))  
    cols.to.zero <- cols.to.zero[-c(1,5, 6)] #compliance, L2 and L3 do not become zero when an individual dies
    # 
    # #columns, used to perform operations on different worm and mf compartments 
     tot.worms <- num.comps.worm*3
     num.cols <- 6 + num.mf.comps + tot.worms 
     worms.start <- 7 + num.mf.comps
    # 
    # 
     nfw.start <- 7 + num.mf.comps + num.comps.worm #start of infertile worms
     fw.end <- num.cols #end of fertile worms 
     mf.start <- 7
     mf.end <- 6 + num.mf.comps
     
    #age-dependent mortality and fecundity rates of parasite life stages

    age.cats <- seq(0, 20, length = num.comps.worm) #up to 20 years old (assume all worms die after age 20 years)

    mort.rates.worms <- weibull.mortality(DT = DT, par1 = mu.w1, par2 = mu.w2, age.cats = age.cats)

    fec.rates.worms <- 1.158305 * fec.w.1 / (fec.w.1 + (fec.w.2 ^ -age.cats) - 1) #no DT - Rk4

    age.cats.mf <- seq(0, 2.5, length = num.mf.comps) #up to 2.5 years old (assume mf die after age 2.5 years)

    #DT not relevent here because RK4 is used to calculate change in mf

    mort.rates.mf <- weibull.mortality(DT = 1, par1 = mu.mf1, par2 = mu.mf2, age.cats = age.cats.mf)

    # #create inital age distribution and simulate stable age distribution
    # 
    # cur.age <- rep(0, N)
    # 
    # #(the approach below must be used, drawing human lifespans from an exponential distribution eventually leads to a non-exponential distribution)
    # for(i in 1 : 75000) #if at equilibrium you saved the age at which inds die and simulated further, you should get an exponential distribution
    # {
    #   cur.age <- cur.age + DT
    #   
    #   death.vec <- rbinom(N, 1, (1/mean.age) * DT) # Matt: human mortality (constant with age) - no. of deaths at time step t is a random variable drawn from binomial distribution (N = human pop size?)
    #   
    #   cur.age[which(death.vec == 1)] <- 0 #set individuals which die to age 0
    #   cur.age[which(cur.age >= real.max.age)] <- 0 #all individuals >= maximum imposed age die (matt: distribution truncated to prevent excessively long life spans - a_max)
    # }
    # 
    # 
    # ex.vec <-rgamma(N, gam.dis, gam.dis) #individual level exposure to fly bites (matt: individual-specific exposure factor assigned at birth - drawn from gamma dist, with shape par (K_E = gam.dis, and rate par set to this))
    # 
    # ###############################################
    # #matrix for delay in L3 establishment in humans 
    # num.delay.cols <- l3.delay * (28 / dt.days) 
    # l.extras <- matrix(0, ncol= num.delay.cols, nrow= N)
    # inds.l.mat <- seq(2,(length(l.extras[1,]))) #for moving columns along with time
    # 
    # ################################################
    # #L1 delay in flies
    # l1.delay <- rep(int.L1, N)
    # 
    # ###############################################
    # #matrix for tracking mf for L1 delay
    # num.mfd.cols <- 4 / dt.days
    # mf.delay <- matrix(int.mf, ncol= num.mfd.cols, nrow= N)
    # inds.mfd.mats <- seq(2,(length(mf.delay[1,])))
    # 
    # ###############################################
    # #matrix for exposure (to fly bites) for L1 delay
    # num.exp.cols <- 4 / dt.days
    # exposure.delay <- matrix(ex.vec, ncol= num.exp.cols, nrow= N)
    # inds.exp.mats <- seq(2,(length(exposure.delay[1,])))  
    # 
    # #matrix for first timestep, contains all parasite values, human age, sex and compliance
    # 
    # #all.mats.temp <- matrix(, nrow=N, ncol=num.cols) # error here? (remove the ,)
    # all.mats.temp <- matrix(nrow=N, ncol=num.cols) # error here? (remove the ,)
    # 
    # all.mats.temp[,  (worms.start) : num.cols] <- int.worms
    # 
    # all.mats.temp[, 4] <- int.L1
    # 
    # all.mats.temp[, 5] <- int.L2
    # 
    # all.mats.temp[, 6] <- int.L3
    # 
    # all.mats.temp[, 7 : (7 + (num.mf.comps-1))] <- int.mf
    # 
    # all.mats.temp[,1] <- rep(0, N) #column used during treatment
    # all.mats.temp[,2] <- cur.age
    # 
    # #assign sex to humans 
    # 
    # sex <- rbinom(N, 1, sex.rat) # matt: randomly assigned (binomal dist) with equal probability (e.g., psi_F = 0.5, psi_m = 0.5)
    # 
    # all.mats.temp[,3] <- sex
    # 
    # #non-compliant people
    # non.comp <- ceiling(N * pnc)
    # out.comp <- rep(0, N)
    # s.comp <- sample(N, non.comp)
    # out.comp[s.comp] <- 1
    # all.mats.temp[,1] <- out.comp
    # 
    # # matt: tracking total worms below (newly added)
    worm.i.track <- list()
    tot.worm.mean <- c()
    tot.worms <- c()
    # nw.rate.track <- c()
    # nw.rate.i.track <- list()
    # 
    # 
    treat.vec.in <- rep(NA, N) #for time since treatment calculations 
    # 
    # prev <-  c()
    # mean.mf.per.snip <- c()
    # 
    i <- 1
  
  # ====================================================================================== #
  # extracting input from CSV (python or R files to feed into loop below; for time t + 1)  #
  
  # while( i == 1) {
  
  # imported #
  all.mats.temp <- all.mats.temp.in
  exposure.delay <- exposure.delay.in 
  ex.vec <- ex.vec.in
  l1.delay <- l1.delay.in
  l.extras <- l.extras.in
  mf.delay <- mf.delay.in
  
  # objects required from imported objects above
  prev <- all.mats.temp[[2]]
  mean.mf.per.snip <- all.mats.temp[[3]]
  
  # other objects required from # code above (& using objects imported in)
  inds.l.mat <- seq(2,(length(l.extras[1,]))) #for moving columns along with time
  inds.mfd.mats <- seq(2,(length(mf.delay[1,])))
  inds.exp.mats <- seq(2,(length(exposure.delay[1,]))) 
  #}
    
  while(i < time.its) #over time
      
    {
      print(paste(round(i * DT, digits = 2), 'yrs', sep = ' '))
      
      #stores mean L3 and adult worms from previous timesteps 
      
      all.mats.cur <- all.mats.temp 
      
      #which individuals will be treated if treatment is given
      if(i >= treat.start) {cov.in <- os.cov(all.dt = all.mats.cur, pncomp = pnc, covrg = treat.prob, N = N)}
      
      #sex and age dependent exposure, mean exposure must be 1, so ABR is meaningful
      
      mls <- which(all.mats.cur[,3] == 1) # matt : ?
      fmls <- which(all.mats.cur[,3] == 0) # matt: ?
      
      s.a.exp <- rep(0, N)
      
      s.a.exp[mls] <- m.exp * exp(-age.exp.m * (all.mats.cur[mls, 2]))
      
      gam.m <- 1 / mean(s.a.exp[mls]) #normalize so mean = 1 (matt: is this equivalent to including the gamma_s term?)
      s.a.exp[mls] <- s.a.exp[mls] * gam.m
      
      s.a.exp[fmls] <- f.exp * exp(-age.exp.f * (all.mats.cur[fmls, 2]))
      
      gam.f <- 1 / mean(s.a.exp[fmls]) #normalize so mean = 1
      s.a.exp[fmls] <- s.a.exp[fmls] * gam.f
      
      ex.vec <- ex.vec * (1 / mean(ex.vec)) #normalize so mean = 1 (matt: normalising the indvidual-specific exposure from line 565)
      
      tot.ex.ai <- s.a.exp * ex.vec # matt: combine sex/age specific exposure + individual specific exposure (total exposure to blackfly bites)
      tot.ex.ai <- tot.ex.ai * (1 / mean(tot.ex.ai)) #normalize so mean = 1
      
      #increase age (for next time step)
      
      all.mats.temp[,2] <- (all.mats.cur[,2]) + DT #increase age for all individuals
      
      death.vec <- rbinom(N, 1, (1/mean.age) * DT) #select individuals to die
      
      to.die <- which(death.vec == 1)
      
      at.ab.max <- which(all.mats.temp[,2] >= real.max.age)
      
      to.die <- c(to.die, at.ab.max)
      
      to.die <- unique(to.die) #may have repeated indivudals i.e selected by binom and >80 
      
      ##################
      #delay calculations 
      ##################
      
      #there is a delay in new parasites entering humans (from fly bites) and entering the first adult worm age class
      
      new.worms.m <- c()
      new.worms.nf <- c()
      
      new.worms.m <- rbinom(N, size = l.extras[,length(l.extras[1,])], prob = 0.5) #draw males and females from last column of delay matrix
      new.worms.nf <- l.extras[,length(l.extras[1,])] - new.worms.m
      
      #move individuals along
      l.extras[,inds.l.mat] <- l.extras[,(inds.l.mat-1)]
      
      #mean number of L3 in fly population
      L3.in <- mean(all.mats.cur[, 6])
      
      # matt - to track below
      worm.i <- rowSums(all.mats.cur[, worms.start:num.cols])
      worm.i.track[[i]] <- rowSums(all.mats.cur[, worms.start:num.cols]) # to track
      tot.worm.mean[i] <- mean(worm.i) # to track
      tot.worms[i] <- sum(worm.i) # to track (sum of sum of all worms per individual)
      
      
      #rate of infections in humans
      #delta.hz, delta.hinf, c.h are density dependence parameters, expos is the exposure of each person to bites
      nw.rate <- Wplus1.rate(delta.hz, delta.hinf, c.h, L3 = L3.in, m ,
                             beta, expos = tot.ex.ai, DT)
      
      
      new.worms <- rpois(N, nw.rate) #total new establishing L3 for each individual 
      
      l.extras[,1] <- new.worms
      
      
      for(k in 1 : num.comps.worm) #go through each adult worm compartment
        
      {
        
        if(k == 1) {from.last <- rep(0, N)} #create vector for worms coming from previous compartment (needs to be 0 when k ==1)
        
        
        res <- change.worm.per.ind(delta.hz = delta.hz, delta.hinf = delta.hinf, c.h = c.h, L3 = L3.in, m = m , beta = beta, compartment = k, 
                                   total.dat = all.mats.cur, num.comps = num.comps.worm,
                                   w.f.l.c = from.last, lambda.zero = lambda.zero, omeg = omeg, expos = tot.ex.ai, 
                                   ws = worms.start, DT = DT, mort.rates = mort.rates.worms, time.each.comp = time.each.comp.worms, new.worms.m = new.worms.m, 
                                   new.worms.nf.fo = new.worms.nf, lam.m = lam.m, phi = phi, treat.stop = treat.stop, iteration = i, treat.int = treat.int, treat.prob = treat.prob, 
                                   cum.infer = cum.infer, treat.vec = treat.vec.in, 
                                   give.treat = give.treat, treat.start = treat.start, N = N, onchosim.cov = cov.in, times.of.treat = times.of.treat.in)
        
        
        
        from.last <- res #assign output to use at next iteration, indexes 2, 5, 6 (worms moving through compartments)
        
        #update male worms in matrix for compartment k
        
        all.mats.temp[, (6 + num.mf.comps + k)] <- res[[1]]
        
        #update females worms in matrix
        
        all.mats.temp[, (6 + num.mf.comps + num.comps.worm + k)] <- res[[3]] #infertile, num.comps.worm skips over males
        all.mats.temp[, (6 + num.mf.comps + 2*num.comps.worm + k)] <- res[[4]] #fertile, num.comps.worm skips over males and infertile females
        
        
      }
      
      if(give.treat == 1 & i >= treat.start) {treat.vec.in <- res[[7]]} #treated individuals
      
      for(mf.c in 1 : num.mf.comps)   
        
      {
        
        res.mf <- change.micro(dat = all.mats.cur, num.comps =num.comps.worm, mf.cpt = mf.c, 
                               num.mf.comps = num.mf.comps, ws=worms.start, DT=DT, time.each.comp = time.each.comp.mf, 
                               mu.rates.mf = mort.rates.mf, fec.rates = fec.rates.worms, mf.move.rate = mf.move.rate, up = up, kap = kap, iteration = i, treat.vec = treat.vec.in, give.treat = give.treat, treat.start = treat.start)
        
        all.mats.temp[, 6 + mf.c] <- res.mf
      }
      
      
      #inputs for delay in L1  
      exp.delay.temp <- exposure.delay[, length(exposure.delay[1,])]
      mf.delay.temp <- mf.delay[, length(mf.delay[1,])]
      l1.delay.temp <- l1.delay #L1 from previous timestep
      
      #move values along
      exposure.delay[, inds.exp.mats] <- exposure.delay[, (inds.exp.mats -1)]
      mf.delay[, inds.mfd.mats] <- mf.delay[, (inds.mfd.mats - 1)] 
      
      #update L1, L2 and L3
      
      #total number of mf in each person
      mf.temp <- rowSums(all.mats.cur[, 7 : (6 + num.mf.comps)]) #sum mf over compartments, mf start in column 7
      
      all.mats.temp[, 4] <- calc.L1(beta, mf = mf.temp, mf.delay.in = mf.delay.temp, expos = tot.ex.ai, delta.vo, c.v, nuone, mu.v, a.v, expos.delay = exp.delay.temp)
      all.mats.temp[, 5] <- calc.L2(nuone, L1.in = l1.delay.temp, mu.v, nutwo, mf = mf.delay.temp, a.v, expos = exp.delay.temp)
      all.mats.temp[, 6] <- calc.L3(nutwo, L2.in = all.mats.cur[, 5], a.H, g, mu.v, sigma.L0)
      
      #new values for delay parts
      l1.delay <- all.mats.temp[, 4]
      mf.delay[, 1] <- rowSums(all.mats.cur[, 7 : (6 + num.mf.comps)])
      exposure.delay[, 1] <- tot.ex.ai
      
      
      #new individual exposure for newborns, clear rows for new borns
      
      if(length(to.die) > 0)
      {
        ex.vec[to.die] <- rgamma(length(to.die), gam.dis, gam.dis)
        
        l.extras[to.die, ] <- 0 #establishing adult worms 
        
        mf.delay[to.die, 1] <- 0 #individual dies so no contribution to L1s at this timestep
        
        l1.delay[to.die] <- 0
        
        treat.vec.in[to.die] <- NA
        
        all.mats.temp[to.die, cols.to.zero] <- 0 #set age, sex and parasites to 0 (includes L1, but not L2 L3)
        all.mats.temp[to.die, 3] <- rbinom(length(to.die), 1, 0.5) #draw sex
      }
      
      temp.mf <- mf.per.skin.snip(ss.wt = 2, num.ss = 2, slope.kmf = 0.0478, int.kMf = 0.313, data = all.mats.temp, nfw.start, fw.end, 
                                  mf.start, mf.end, pop.size = N)
      
      prev <-  c(prev, prevalence.for.age(age = min.mont.age, ss.in = temp.mf, main.dat = all.mats.temp))
      
      
      mean.mf.per.snip <- c(mean.mf.per.snip, mean(temp.mf[[2]][which(all.mats.temp[,2] >= min.mont.age)]))
      
      mf.per.skin.snp.out <- temp.mf[[2]] #to extract out mf per skin snip for each individual?
      
      current_i <- i
      
      i <- i + 1
      
      time <- current_i/366
      
      updated_time <- (8 - 1/366) + current_i/366 # time fromm imported run 
      
      
      list[[i]] <-list(all.mats.temp, prev, mean.mf.per.snip, mf.per.skin.snp.out,
                       tot.worm.mean, tot.worms, worm.i.track,
                       ex.vec, mf.delay, exposure.delay, l1.delay, l.extras,
                       current_i, i, time, updated_time) #[[2]] is mf prevalence, [[3]] is intensity
      # tot.worm.mean, tot.worms, worm.i.track not correct as drawn from empty vectors/lists at time i
      
    }
    

  return(list)
  
}

# ==================================== #
# specify timesteps and model duration #

DT.in <- 1/366 #timestep must be one day

# ABR.in <- 1000

treat.len <- 0 #treatment duration in years

#treat.strt  = round(40 / (DT.in ))

#treat.strt  = round((8 + 1/ 366 + 1/ 366)/ (DT.in ))

treat.strt  = round((1/ 366 + 1/ 366)/ (DT.in ))

treat.stp = treat.strt + round(treat.len / (DT.in )) #treatment start and stop


# note if not treatment, then this number is total model run time!

gv.trt = 0 # IVM given so 1
trt.int = 0 #treatment interval (years, 0.5 gives biannual, 1 is annual)

# iter_seq <- seq(1 , treat.strt, by = 1) # if want to extract list for all iterations in a run

# i_to_select <- c(treat.strt -1, treat.strt) # if want to extract list for specific iterations

# ======================================================================================== #
#   import in model objects to feed into R model (this is t - 1; so 8 years - dt step)     #

allmatstemp_t.1 <- read.csv("R_data_files/allmatstemp_t-1.csv")
allmatstemp_t.1 <- allmatstemp_t.1[,-1]
all.mats.temp.in <- data.matrix(allmatstemp_t.1)

exposuredelay_t.1 <- read.csv("R_data_files/exposuredelay_t-1.csv")
exposuredelay_t.1 <- exposuredelay_t.1[,-1]
exposure.delay.in <- data.matrix(exposuredelay_t.1)

exvec_t.1 <- read.csv("R_data_files/exvec_t-1.csv")
ex.vec.in <- exvec_t.1[,-1]

l1edelay_t.1 <- read.csv("R_data_files/l1edelay_t-1.csv")
l1.delay.in <- l1edelay_t.1[,-1]

lextras_t.1 <- read.csv("R_data_files/lextras_t-1.csv")
lextras_t.1 <- lextras_t.1[,-1]
l.extras.in <- matrix(as.numeric(unlist(lextras_t.1)),nrow=nrow(lextras_t.1)) # converts to a large numeric matrix correctly
#l.extras.in <- data.matrix(lextras_t.1)

mfdelay_t.1 <- read.csv("R_data_files/mfdelay_t-1.csv")
mfdelay_t.1 <- mfdelay_t.1[,-1]
mf.delay.in <- data.matrix(mfdelay_t.1)

 # now run model with imported objects #
timesteps = 2

output <-  ep.equi.sim(time.its = timesteps,
                       ABR = 294,
                       DT = DT.in,
                       treat.int = trt.int,
                       treat.prob = 0.65,
                       give.treat = gv.trt,
                       treat.start = treat.strt,
                       treat.stop = treat.stp,
                       pnc = 0.05, min.mont.age = 5,
                       delta.hz.in = 0.186, # # change based on fitted delta_h_0 (corresponding k_E)?
                       delta.hinf.in = 0.003, # change based on fitted delta_h_inf (corresponding k_E)?
                       c.h.in = 0.005,  # change based on fitted cH (corresponding k_E)?
                       gam.dis.in = 0.3, #individual level exposure heterogeneity (matt: shape par in gamma dist, K_E = 0.3 estimated in Hamley et al. 2019)
                       all.mats.temp.in = all.mats.temp.in,
                       exposure.delay.in = exposure.delay.in,
                       ex.vec.in = ex.vec.in,
                       l1.delay.in = l1.delay.in,
                       l.extras.in = l.extras.in,
                       mf.delay.in = mf.delay.in
                       
)
