

#EPIONCHO-IBM
#30/04/2020
#Jonathan Hamley 

#individual-based at the level of humans (but not parasites), age-structued for adult O.volvulus and microfilariae
#parasite life stages in black flies are deterministic and population-based
#all functions are called in ep.equi.sim which gives the final output


#function to calculate which individuals will be treated
#all.dt is a matrix containing (among other things) information on compliance to treatment (whether or not an individual can be treated) # nolint
#and the age of each individual 
#covrg is the total coverage of the population




#function to rotate matrix, used in mf function 
#rotate <- function(x) t(apply(x, 2, rev))

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

#proportion of mf per mg developing into infective larvae within the vector

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




#people are tested for the presence of mf using a skin snip, we assume mf are overdispersed in the skin
#function calculates number of mf in skin snip for all people
#ss.wt is the weight of the skin snip
mf.per.skin.snip <- function(ss.wt, num.ss, slope.kmf, int.kMf, data, nfw.start, fw.end,  ###check vectorization 
                             mf.start, mf.end, pop.size)
  
{
  
  all.mfobs <- c()
  
  kmf <- slope.kmf * (rowSums(data[,nfw.start:fw.end])) + int.kMf #rowSums(da... sums up adult worms for all individuals giving a vector of kmfs
  
  mfobs <- rnbinom(pop.size, size = kmf, mu = ss.wt * (rowSums(data[,mf.start:mf.end])))
  
  nans <- which(mfobs == 'NaN'); mfobs[nans] <- 0
  
  if(num.ss > 1)
    
  {
    
    tot.ss.mf <- matrix(, nrow = length(data[,1]), ncol = num.ss)
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
                        min.mont.age)
  
  
{ 
  
  #hard coded parms


  up = 0.0096; kap = 1.25 #effects of ivermectin
  E0 = 0; q = 0;  #age-dependent exposure to fly bites
  

  
  #columns to set to zero when an individual dies
  cols.to.zero <- seq(from = 1, to = (6 + num.mf.comps + 3*num.comps.worm))  
  cols.to.zero <- cols.to.zero[-c(1,5, 6)] #compliance, L2 and L3 do not become zero when an individual dies
  
  #columns, used to perform operations on different worm and mf compartments 
  tot.worms <- num.comps.worm*3
  num.cols <- 6 + num.mf.comps + tot.worms 
  worms.start <- 7 + num.mf.comps
  
  
  nfw.start <- 7 + num.mf.comps + num.comps.worm #start of infertile worms
  fw.end <- num.cols #end of fertile worms 
  mf.start <- 7
  mf.end <- 6 + num.mf.comps
  
  #age-dependent mortality and fecundity rates of parasite life stages 

  ################################################
  #L1 delay in flies
  l1.delay <- rep(int.L1, N)
  
  ###############################################
  #matrix for tracking mf for L1 delay
  num.mfd.cols <- 4 / dt.days
  mf.delay <- matrix(int.mf, ncol= num.mfd.cols, nrow= N)
  inds.mfd.mats <- seq(2,(length(mf.delay[1,])))
  
  ###############################################
  #matrix for exposure (to fly bites) for L1 delay
  num.exp.cols <- 4 / dt.days
  exposure.delay <- matrix(ex.vec, ncol= num.exp.cols, nrow= N)
  inds.exp.mats <- seq(2,(length(exposure.delay[1,])))  
  
  #matrix for first timestep, contains all parasite values, human age, sex and compliance
  all.mats.temp <- matrix(, nrow=N, ncol=num.cols)
  
  all.mats.temp[,  (worms.start) : num.cols] <- int.worms

  
  all.mats.temp[, 7 : (7 + (num.mf.comps-1))] <- int.mf
  
  all.mats.temp[,1] <- rep(0, N) #column used during treatment
  all.mats.temp[,2] <- cur.age
  
  #assign sex to humans 

  all.mats.temp[,3] <- sex
  
  #non-compliant people
  non.comp <- ceiling(N * pnc)
  out.comp <- rep(0, N)
  s.comp <- sample(N, non.comp)
  out.comp[s.comp] <- 1
  all.mats.temp[,1] <- out.comp
  
  treat.vec.in <- rep(NA, N) #for time since treatment calculations 
  
  prev <-  c()
  mean.mf.per.snip <- c()
  
  i <- 1
  
  while(i < time.its) #over time
    
  {
    #stores mean L3 and adult worms from previous timesteps 
    
    all.mats.cur <- all.mats.temp 


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
    
    
    i <- i + 1
    
  }
  
  return(list(all.mats.temp, prev, mean.mf.per.snip)) #[[2]] is mf prevalence, [[3]] is intensity
  
  
}


DT.in <- 1/366 #timestep must be one day 

treat.len <- 8 #treatment duration in years

treat.strt  = round(25 / (DT.in )); treat.stp = treat.strt + round(treat.len / (DT.in )) #treatment start and stop
timesteps = treat.stp + round(3 / (DT.in )) #final duration

gv.trt = 1
trt.int = 1 #treatment interval (years, 0.5 gives biannual)


ABR.in <- 1000 #annual biting rate 
  
output <-  ep.equi.sim(time.its = timesteps,
              ABR = ABR.in,
              DT = DT.in,
              treat.int = trt.int,
              treat.prob = 65,
              give.treat = gv.trt,
              treat.start = treat.strt,
              treat.stop = treat.stp,
              pnc = 0.05, min.mont.age = 5)
  


