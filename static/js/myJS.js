const passedData = dataFunc(); 


passedData.map((val)=>{
    const data = []; 

    for (let index=0; index<val["X"].length;index++)
    {
        const obj = {};
        obj.x=val["X"][index]; 
        obj.y = val["Y"][index];
        data.push(obj) 
    }
    const ctx = document.getElementsByClassName(val['index'])[0];
    const myLineChart = new Chart(ctx, {
        type: 'line',
        data:{
            datasets: [{
                label:val['label'],
                backgroundColor: null,
                borderColor:  val['color'],
                data,
                fill: false
            }]
        },
        options:{
            responsive: true, 
            scales: {
                xAxes: [{
                      type: 'linear', 
                 }]
           }
        }
    });

})
 
