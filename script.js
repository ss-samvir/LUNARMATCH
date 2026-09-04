const fileA=document.getElementById("fileA"),fileB=document.getElementById("fileB");

function setup(input,img,drop){
  input.addEventListener("change",()=>{
    const f=input.files[0]; if(!f)return;
    img.src=URL.createObjectURL(f); img.style.display="block";
    drop.querySelector(".upload-icon").style.display="none";
    drop.querySelector("strong").style.display="none";
    drop.querySelector("small").style.display="none";
  });
}
setup(fileA,document.getElementById("previewA"),document.querySelectorAll(".drop")[0]);
setup(fileB,document.getElementById("previewB"),document.querySelectorAll(".drop")[1]);

async function compareImages(){
  if(!fileA.files[0]||!fileB.files[0]){
    alert("Please upload both Image A and Image B first.");
    return;
  }

  const btn=document.getElementById("compareBtn");
  const status=document.getElementById("status");
  btn.disabled=true;
  btn.textContent="Analyzing lunar images…";
  status.textContent="ANALYZING…";
  document.getElementById("score").textContent="…";

  const fd=new FormData();
  fd.append("image1",fileA.files[0]);
  fd.append("image2",fileB.files[0]);

  try{
    const response=await fetch("/api/match",{method:"POST",body:fd});
    const data=await response.json();
    if(!response.ok) throw new Error(data.error||"Backend analysis failed.");

    status.textContent=data.match_found ? "MATCH FOUND" : "NO STRONG MATCH";
    document.getElementById("score").textContent=data.match_percentage.toFixed(1)+"%";
    document.getElementById("features").textContent=data.corresponding_features;
    document.getElementById("confidence").textContent=data.confidence;
    document.getElementById("quality").textContent=data.quality;
    document.getElementById("time").textContent=data.processing_time.toFixed(2)+" sec";
    document.getElementById("resultA").src="data:image/jpeg;base64,"+data.visualization;
    document.getElementById("resultB").src="data:image/jpeg;base64,"+data.visualization;
  }catch(err){
    alert(err.message);
    status.textContent="ANALYSIS FAILED";
    document.getElementById("score").textContent="—";
  }finally{
    btn.disabled=false;
    btn.textContent="⌕  COMPARE IMAGES";
  }
}
document.getElementById("compareBtn").addEventListener("click",compareImages);
