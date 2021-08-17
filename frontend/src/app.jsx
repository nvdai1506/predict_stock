import React, { Component } from 'react'
import { Line } from "react-chartjs-2"
import Select from '@material-ui/core/Select'
import MenuItem from '@material-ui/core/MenuItem';
import Grid from '@material-ui/core/Grid';
import InputLabel from '@material-ui/core/InputLabel';

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
    this.state = { model: "LSTM" }
    this.chartReference = React.createRef();
    this.chartReferencePoc = React.createRef();
  }

  callApi(model) {
    fetch(`http://localhost:5000/model?model=${model}`, {
      method: "GET",
      crossDomain: true,

    }).then(response => response.json())
      .then(data => this.updateData(data)).catch(error => console.error(error));
  }

  updateData(rawData) {
    console.log(rawData);
    var datasets = [{
      label: "Price predictions",
      borderColor: 'rgb(75, 192, 192)',
      data: []
    }, {
      label: "Close Price",
      borderColor: 'rgb(75, 192, 0)',
      data: []
    }];
    var indexs = rawData['Index'];
    var predictions = rawData['Predictions'];
    var closes = rawData['Close'];
    var labels = [];
    indexs.forEach((element, index) => {
      labels.push(new Date(element));
      datasets[0].data.push({ x: new Date(element), y: predictions[index] });
      datasets[1].data.push({ x: new Date(element), y: closes[index] });
    });
    this.chartReference.current.data.labels = labels;
    this.chartReference.current.data.datasets = datasets;
    this.chartReference.current.update();

    var datasets_poc = [{
      label: "Price of change",
      borderColor: 'rgb(75, 192, 192)',
      data: []
    }];
    var indexs_poc = rawData['IndexPoC'];
    var pocs = rawData['poc'];
    var labels_poc = [];
    indexs_poc.forEach((element, index) => {
      labels_poc.push(new Date(element));
      datasets_poc[0].data.push({ x: new Date(element), y: pocs[index] });
    });
    this.chartReferencePoc.current.data.labels = labels_poc;
    this.chartReferencePoc.current.data.datasets = datasets_poc;
    this.chartReferencePoc.current.update();
  }

  componentDidMount() {
    this.callApi(this.state.model);
  }

  changeModel(value) {
    console.log(value);
    if (value != this.state.model) {
      this.callApi(value);
    }
    this.setState({ model: value });
  }

  render() {
    return (
      <div>
        <Grid
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
        >
          <Grid item><h1>Stock Price Analysis Dashboard1 ☘️</h1></Grid>
          <Grid item><InputLabel id="select-model">Model</InputLabel>
            <Select
              labelId="select-model"
              id="select-model"
              value={this.state.model}
              onChange={(event) => this.changeModel(event.target.value)}
            >
              <MenuItem value={'XGBoost'}>XGBoost</MenuItem>
              <MenuItem value={'RNN'}>RNN</MenuItem>
              <MenuItem value={'LSTM'}>LSTM</MenuItem>
              <MenuItem value={'SVR'}>SVR</MenuItem>
              <MenuItem value={'Linear'}>Linear</MenuItem>
            </Select>
          </Grid>
        </Grid>
        <Line
          ref={this.chartReference}
          id="main"
          options={options}
        />
        <Grid
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
        >
          <Grid item><h1>Price of Change</h1></Grid>
          <Line
            ref={this.chartReferencePoc}
            id="poc"
            options={options}
          />

        </Grid>
      </div>

    )
  }
}

export default App