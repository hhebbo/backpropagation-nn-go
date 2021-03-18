package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

type dataset struct {
	className string
	data      []float64
	class     float64
}

const numOfInputPoints = 25
const numOfHiddenNeurons = 15
const numOfOutputNeurons = 1
const learningRate = 0.5
const rmsThreshold = 0.00001

func getDataSet() []dataset {

	return []dataset{
		{"A", []float64{0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0}, 0.1},
		{"R", []float64{0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0}, 0.2},
		{"N", []float64{0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0}, 0.3},
		{"H", []float64{0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0}, 0.4},
		{"S", []float64{0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0}, 0.5},
		{"O", []float64{0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0}, 0.6},
		{"D", []float64{0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0}, 0.7},
		{"F", []float64{0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0}, 0.8},
		{"T", []float64{0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}, 0.9},
		{"L", []float64{0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0}, 1.0},
	}
}

func main() {
	switch os.Args[1:][0] {
	case "learn":
		learn()
	case "predict":
		inputSample := os.Args[1:][1]
		var sample []float64

		inputSampleSplit := strings.Split(inputSample, ",")
		sample = make([]float64, len(inputSampleSplit), len(inputSampleSplit))
		for i := 0; i < len(inputSampleSplit); i += 1 {
			f64, _ := strconv.ParseFloat(inputSampleSplit[i], 64)
			sample[i] = float64(f64)
		}

		predict(sample)
	default:
		fmt.Println("Please choose either: learn or predict []float64")
	}
}

func learn() ([][]float64, [][]float64) {
	hiddenWeights := initiateWeights(numOfHiddenNeurons, numOfInputPoints)
	outputWeights := initiateWeights(numOfOutputNeurons, numOfHiddenNeurons)

	// feed forward
	data := getDataSet()
	epoch := 0
	numOfObservedSamples := 0.0
	sumOfErrors := [numOfOutputNeurons]float64{}

	for i := 0; i < len(data); i++ {
		sample := data[i]

		hiddenNeurons := activateNeurons(numOfHiddenNeurons, numOfInputPoints, sample.data, hiddenWeights)
		outputNeurons := activateNeurons(numOfOutputNeurons, numOfHiddenNeurons, hiddenNeurons, outputWeights)

		// calculate error
		var outputErrors = make([]float64, numOfOutputNeurons)
		var dOutputErrors = make([]float64, numOfOutputNeurons)
		var tOutputErrors = make([]float64, numOfOutputNeurons)
		for j := 0; j < numOfOutputNeurons; j++ {
			outputErrors[j] = outputNeurons[j] - sample.class
			dOutputErrors[j] = outputNeurons[j] * (1 - outputNeurons[j])
			tOutputErrors[j] = outputErrors[j] * dOutputErrors[j]
			fmt.Println(sample.class)
		}

		var hiddenErrors = make([]float64, numOfHiddenNeurons)
		var dHiddenErrors = make([]float64, numOfHiddenNeurons)
		var tHiddenErrors = make([]float64, numOfHiddenNeurons)
		for j := 0; j < numOfOutputNeurons; j++ {
			for k := 0; k < numOfOutputNeurons; k++ {
				hiddenErrors[j] = tOutputErrors[k] * outputWeights[k][j]
				dHiddenErrors[j] = hiddenNeurons[j] * (1 - hiddenNeurons[j])
				tHiddenErrors[j] = hiddenErrors[j] * dHiddenErrors[j]
			}
		}

		// rms
		rms := 0.0
		numOfObservedSamples++
		for j := 0; j < numOfOutputNeurons; j++ {
			sumOfErrors[j] += outputErrors[j]
			rms += math.Pow(sumOfErrors[j], 2.0)
		}
		rms = math.Sqrt(rms) / numOfObservedSamples

		// is learning done?
		if rms < rmsThreshold {
			printFinalOutput(hiddenWeights, outputWeights, epoch)

			saveWeights(hiddenWeights, "weights_hidden.txt")
			saveWeights(outputWeights, "weights_output.txt")

			return hiddenWeights, outputWeights
		}

		// otherwise update the weights, and continue learning
		updateWeights(numOfHiddenNeurons, numOfInputPoints, tHiddenErrors, sample.data, hiddenWeights)
		updateWeights(numOfOutputNeurons, numOfHiddenNeurons, tOutputErrors, hiddenNeurons, outputWeights)

		if i+1 == len(data) {
			i = -1
			epoch++
		}
	}

	return hiddenWeights, outputWeights
}

func predict(sampleData []float64) {
	hiddenWeights, _ := readWeights("weights_hidden.txt", numOfHiddenNeurons)
	outputWeights, _ := readWeights("weights_output.txt", numOfOutputNeurons)

	hiddenNeurons := activateNeurons(numOfHiddenNeurons, numOfInputPoints, sampleData, hiddenWeights)
	outputNeurons := activateNeurons(numOfOutputNeurons, numOfHiddenNeurons, hiddenNeurons, outputWeights)

	var outputResults = make([]float64, numOfOutputNeurons)
	for j := 0; j < numOfOutputNeurons; j++ {
		outputResults[j] = math.Round(outputNeurons[j]*10) / 10
	}

	data := getDataSet()
	var finalResult string
	for i := 0; i < len(data); i++ {
		for j := 0; j < numOfOutputNeurons; j++ {
			if data[i].class == outputResults[j] {
				finalResult = data[i].className
			}
		}
	}

	fmt.Println("----------------------------------------------------------------------------------------------------------------")
	fmt.Println("-------Result-------")
	fmt.Println("Neural network output is: ", outputNeurons)
	fmt.Println("This is: ", finalResult)
	fmt.Println()
}

func initiateWeights(layerDim int, prevLayerDin int) [][]float64 {
	var layerWeights = make([][]float64, layerDim)

	rand.Seed(time.Now().UnixNano())
	for i := 0; i < layerDim; i++ {
		layerWeights[i] = make([]float64, prevLayerDin)
		for j := 0; j < prevLayerDin; j++ {
			layerWeights[i][j] = rand.Float64()
		}
	}

	return layerWeights
}

func activateNeurons(layerDim int, prevLayerDin int, input []float64, weights [][]float64) []float64 {
	var neurons = make([]float64, layerDim)

	for j := 0; j < layerDim; j++ {
		sigma := 0.0
		for k := 0; k < prevLayerDin; k++ {
			sigma += input[k] * weights[j][k]
		}

		neurons[j] = sigmoid(sigma)
	}

	return neurons
}

func updateWeights(layerDim int, prevLayerDin int, tErrors []float64, input []float64, weights [][]float64) {
	for j := 0; j < layerDim; j++ {
		for k := 0; k < prevLayerDin; k++ {
			hdw := tErrors[j] * input[k]
			weights[j][k] = weights[j][k] - (learningRate * hdw)
		}
	}
}

func sigmoid(sigma float64) float64 {
	return (1.0 / (1.0 + math.Exp(-sigma)))
}

func printFinalOutput(hiddenWeights [][]float64, outputWeights [][]float64, epoch int) {
	fmt.Println("----------------------------------------------------------------------------------------------------------------")

	fmt.Println("Yay, I'm done!!!")
	fmt.Println()

	fmt.Println("-------Number of calculations-------")
	fmt.Println(epoch * (numOfInputPoints*numOfHiddenNeurons + numOfHiddenNeurons*numOfOutputNeurons))
	fmt.Println()

	fmt.Println("-------Number of epochs-------")
	fmt.Println(epoch)
	fmt.Println()

	fmt.Println("-------Final hidden weights-------")
	for j := 0; j < numOfHiddenNeurons; j++ {
		fmt.Print("Hidden neuron ", j+1, ": [ ")
		for k := 0; k < numOfInputPoints; k++ {
			fmt.Print(hiddenWeights[j][k], " ")
		}
		fmt.Println("]")
		fmt.Println()
	}

	fmt.Println("-------Final output weights-------")
	for j := 0; j < numOfOutputNeurons; j++ {
		fmt.Print("Output neuron ", j+1, ": [ ")
		for k := 0; k < numOfHiddenNeurons; k++ {
			fmt.Print(outputWeights[j][k], " ")
		}
		fmt.Println("]")
		fmt.Println()
	}

	fmt.Println("----------------------------------------------------------------------------------------------------------------")
}

func saveWeights(weights [][]float64, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	w := bufio.NewWriter(file)
	for _, line := range weights {
		fmt.Fprintln(w, line)
	}

	return w.Flush()
}

func readWeights(path string, layerDim int) ([][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var weights = make([][]float64, layerDim)
	index := 0
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		weightString := scanner.Text()
		weightString = strings.ReplaceAll(weightString, "[", "")
		weightString = strings.ReplaceAll(weightString, "]", "")
		weightStringSplit := strings.Split(weightString, " ")
		weightSlice := make([]float64, len(weightStringSplit), len(weightStringSplit))
		for i := 0; i < len(weightStringSplit); i += 1 {
			f64, _ := strconv.ParseFloat(weightStringSplit[i], 64)
			weightSlice[i] = float64(f64)
		}

		weights[index] = weightSlice
		index++
	}

	return weights, scanner.Err()
}
