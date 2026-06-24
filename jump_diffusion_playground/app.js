
const world=document.getElementById('world');
const ctx=world.getContext('2d');
const W=world.width,H=world.height;

let N=500;
let particles=[];
let running=true;
let survival=[];
let step=0;

function randn(){
 let u=Math.random(),v=Math.random();
 return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}

function initRandom(){
 particles=[];
 for(let i=0;i<N;i++){
  particles.push({x:Math.random()*W,y:H/2+(Math.random()-0.5)*120,alive:true});
 }
}

function spawnExperiment(){
 particles=[];
 for(let i=0;i<N;i++){
  particles.push({x:W*0.25,y:H/2+(Math.random()-0.5)*120,alive:true});
 }
}

function update(){
 const sigma=parseFloat(document.getElementById('sigma').value);
 const lambda=parseFloat(document.getElementById('lambda').value);
 const jumpSize=parseFloat(document.getElementById('jumpSize').value);

 let alive=0;
 for(const p of particles){
  if(!p.alive) continue;

  p.x += sigma*randn();

  if(Math.random()<lambda){
    p.x += jumpSize*randn();
  }

  if(p.x<0 || p.x>W){
    p.alive=false;
  } else {
    alive++;
  }
 }
 survival.push(alive/N);
 if(survival.length>200) survival.shift();
 document.getElementById('stats').innerHTML =
   `Alive: ${alive} / ${N} | Step ${step}`;
 step++;
 updateCharts();
}

function draw(){
 ctx.clearRect(0,0,W,H);
 for(const p of particles){
  if(!p.alive) continue;
  ctx.beginPath();
  ctx.arc(p.x,p.y,3,0,Math.PI*2);
  ctx.fillStyle='cyan';
  ctx.fill();
 }
}

const histChart=new Chart(document.getElementById('hist'),{
 type:'bar',
 data:{labels:[],datasets:[{label:'Density',data:[]}]},
 options:{animation:false}
});

const survChart=new Chart(document.getElementById('survival'),{
 type:'line',
 data:{labels:[],datasets:[{label:'Survival',data:[]}]},
 options:{animation:false}
});

const termChart=new Chart(document.getElementById('terminal'),{
 type:'bar',
 data:{labels:[],datasets:[{label:'Terminal Density',data:[]}]},
 options:{animation:false}
});

function histogram(values,bins=25){
 let h=new Array(bins).fill(0);
 values.forEach(v=>{
  let i=Math.floor(v/W*bins);
  if(i>=0 && i<bins) h[i]++;
 });
 return h;
}

function updateCharts(){
 let xs=particles.filter(p=>p.alive).map(p=>p.x);
 let h=histogram(xs);

 histChart.data.labels=h.map((_,i)=>i);
 histChart.data.datasets[0].data=h;
 histChart.update();

 survChart.data.labels=survival.map((_,i)=>i);
 survChart.data.datasets[0].data=survival;
 survChart.update();

 termChart.data.labels=h.map((_,i)=>i);
 termChart.data.datasets[0].data=h;
 termChart.update();
}

function loop(){
 if(running){
  update();
  draw();
 }
 requestAnimationFrame(loop);
}

function startSim(){running=true;}
function pauseSim(){running=false;}
function resetSim(){
 survival=[]; step=0; initRandom();
}

initRandom();
loop();
