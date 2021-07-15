import React, { Component } from 'react'
import { Line } from "react-chartjs-2"

const options = {
  legend: { display: false },
  title: {
    display: true,
    text: "Predicted world population (millions) in 2050"
  },
  scales: {
    xAxes: [{
      title: "time",
      type: 'time',
      gridLines: {
        lineWidth: 2
      },
      time: {
        unit: 'year'
      }
    }]
  }
}
class App extends Component {
  constructor(props) {
    super(props)
    this.state = { data: [] }
    this.chartReference = React.createRef();
  }

  callApi() {
    fetch("http://localhost:5000/model?model=LSTM", {
      method: "GET",
      crossDomain: true,

    }).then(response => response.json())
    .then(data => this.updateData(data)).catch(error => console.error(error));
  }

  updateData(rawData) {
    this.chartReference.current.data = {
      labels: [],
      datasets: [
        {
          label: "Price",
          borderColor: 'rgb(75, 192, 192)',
          data: []
        }
      ]
    };

    var labels = []
    var data = []
    rawData.forEach(element => {
      labels.push(new Date(element[0]));
      data.push({x: new Date(element[0]), y: element[1]})
    });
    this.chartReference.current.data.labels = labels;
    this.chartReference.current.data.datasets[0].data = data;
    this.chartReference.current.update();
  }

  componentDidMount() {
    this.callApi();
  }

  render() {
    return (
      <h1>
        Stock Price Analysis Dashboard1 ☘️ MNT
        <p></p>
        <Line
          ref={this.chartReference}
          id="main"
          options={options}
        />
      </h1>

    )
  }
}

export default App