package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const numFeatures = 8

type DailyRecord struct {
	Date        time.Time
	Temp        float64
	TempMin     float64
	TempMax     float64
	FeelsLike   float64
	DewPoint    float64
	Pressure    float64
	Humidity    float64
	WindSpeed   float64
	Clouds      float64
	WeatherMain string
}

type LinearModel struct {
	Weights [numFeatures]float64
	Bias    float64
}

type WeatherPredictor struct {
	TempModel      LinearModel
	TempMinModel   LinearModel
	TempMaxModel   LinearModel
	FeelsLikeModel LinearModel
	HumidityModel  LinearModel
	PressureModel  LinearModel
	WindModel      LinearModel

	FeatureMean [numFeatures]float64
	FeatureStd  [numFeatures]float64

	TempMean, TempStd           float64
	TempMinMean, TempMinStd     float64
	TempMaxMean, TempMaxStd     float64
	FeelsLikeMean, FeelsLikeStd float64
	HumidityMean, HumidityStd   float64
	PressureMean, PressureStd   float64
	WindMean, WindStd           float64

	DailyData []DailyRecord

	// month -> tempBucket -> weather_main -> count
	WeatherStats map[int]map[int]map[string]int
}

type Prediction struct {
	Date        string  `json:"date"`
	City        string  `json:"city"`
	TempAvg     float64 `json:"temp_avg_c"`
	TempMin     float64 `json:"temp_min_c"`
	TempMax     float64 `json:"temp_max_c"`
	FeelsLike   float64 `json:"feels_like_c"`
	Humidity    float64 `json:"humidity_pct"`
	WindSpeed   float64 `json:"wind_speed_ms"`
	Pressure    float64 `json:"pressure_hpa"`
	WeatherMain string  `json:"weather_main"`
	WeatherDesc string  `json:"weather_description"`
}

func loadAndAggregate(filePath string) ([]DailyRecord, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	type accumulator struct {
		tempSum, dewPointSum, feelsLikeSum     float64
		pressureSum, humiditySum, windSpeedSum float64
		cloudsSum                              float64
		tempMin, tempMax                       float64
		weatherCounts                          map[string]int
		count                                  int
		date                                   time.Time
	}

	dailyMap := make(map[string]*accumulator)
	var dateOrder []string

	for i := 1; i < len(records); i++ {
		row := records[i]
		if len(row) < 15 {
			continue
		}

		dateStr := strings.Split(row[1], " ")[0]
		temp, _ := strconv.ParseFloat(row[2], 64)
		dewPoint, _ := strconv.ParseFloat(row[3], 64)
		feelsLike, _ := strconv.ParseFloat(row[4], 64)
		pressure, _ := strconv.ParseFloat(row[7], 64)
		humidity, _ := strconv.ParseFloat(row[8], 64)
		windSpeed, _ := strconv.ParseFloat(row[9], 64)
		clouds, _ := strconv.ParseFloat(row[11], 64)
		weatherMain := row[13]

		acc, exists := dailyMap[dateStr]
		if !exists {
			t, _ := time.Parse("2006-01-02", dateStr)
			acc = &accumulator{
				weatherCounts: make(map[string]int),
				date:          t,
				tempMin:       temp,
				tempMax:       temp,
			}
			dailyMap[dateStr] = acc
			dateOrder = append(dateOrder, dateStr)
		}

		acc.tempSum += temp
		acc.dewPointSum += dewPoint
		acc.feelsLikeSum += feelsLike
		acc.pressureSum += pressure
		acc.humiditySum += humidity
		acc.windSpeedSum += windSpeed
		acc.cloudsSum += clouds
		acc.weatherCounts[weatherMain]++
		if temp < acc.tempMin {
			acc.tempMin = temp
		}
		if temp > acc.tempMax {
			acc.tempMax = temp
		}
		acc.count++
	}

	days := make([]DailyRecord, 0, len(dateOrder))
	for _, dateStr := range dateOrder {
		acc := dailyMap[dateStr]
		n := float64(acc.count)

		dominant := ""
		maxCount := 0
		for w, c := range acc.weatherCounts {
			if c > maxCount {
				maxCount = c
				dominant = w
			}
		}

		days = append(days, DailyRecord{
			Date:        acc.date,
			Temp:        acc.tempSum / n,
			TempMin:     acc.tempMin,
			TempMax:     acc.tempMax,
			FeelsLike:   acc.feelsLikeSum / n,
			DewPoint:    acc.dewPointSum / n,
			Pressure:    acc.pressureSum / n,
			Humidity:    acc.humiditySum / n,
			WindSpeed:   acc.windSpeedSum / n,
			Clouds:      acc.cloudsSum / n,
			WeatherMain: dominant,
		})
	}

	return days, nil
}

func extractFeatures(day DailyRecord) [numFeatures]float64 {
	doy := float64(day.Date.YearDay())
	return [numFeatures]float64{
		math.Sin(2 * math.Pi * doy / 365.0),
		math.Cos(2 * math.Pi * doy / 365.0),
		day.Temp,
		day.DewPoint,
		day.Pressure,
		day.Humidity,
		day.WindSpeed,
		day.Clouds,
	}
}

func meanStd(data []float64) (float64, float64) {
	n := float64(len(data))
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= n

	variance := 0.0
	for _, v := range data {
		d := v - mean
		variance += d * d
	}
	std := math.Sqrt(variance / n)
	if std < 1e-8 {
		std = 1.0
	}
	return mean, std
}

func trainLinearRegression(
	features [][numFeatures]float64,
	targets []float64,
	fMean, fStd [numFeatures]float64,
	tMean, tStd float64,
	epochs int,
	lr float64,
) LinearModel {
	n := len(features)

	// Normalize features and targets; append bias column of 1s to X
	cols := numFeatures + 1
	normF := make([]float64, n*cols)
	normT := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < numFeatures; j++ {
			normF[i*cols+j] = (features[i][j] - fMean[j]) / fStd[j]
		}
		normF[i*cols+numFeatures] = 1.0 // bias column
		normT[i] = (targets[i] - tMean) / tStd
	}

	// Build computation graph
	g := gorgonia.NewGraph()

	xT := tensor.New(tensor.WithShape(n, cols), tensor.WithBacking(normF))
	yT := tensor.New(tensor.WithShape(n, 1), tensor.WithBacking(normT))

	xNode := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("X"))
	yNode := gorgonia.NodeFromAny(g, yT, gorgonia.WithName("y"))

	// Weights include bias as the last element
	w := gorgonia.NewMatrix(g, gorgonia.Float64,
		gorgonia.WithShape(cols, 1),
		gorgonia.WithName("w"),
		gorgonia.WithInit(gorgonia.Zeroes()))

	// Forward pass: pred = X @ w (bias absorbed into w via ones column)
	pred := gorgonia.Must(gorgonia.Mul(xNode, w))

	// Loss: mean((pred - y)^2)
	diff := gorgonia.Must(gorgonia.Sub(pred, yNode))
	sq := gorgonia.Must(gorgonia.Square(diff))
	loss := gorgonia.Must(gorgonia.Mean(sq))

	// Automatic differentiation
	if _, err := gorgonia.Grad(loss, w); err != nil {
		log.Fatalf("Failed to compute gradients: %v", err)
	}

	// Train
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(w))
	defer vm.Close()

	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(lr))

	for i := 0; i < epochs; i++ {
		if err := vm.RunAll(); err != nil {
			log.Fatalf("Training error at epoch %d: %v", i, err)
		}
		if err := solver.Step(gorgonia.NodesToValueGrads(gorgonia.Nodes{w})); err != nil {
			log.Fatalf("Solver error: %v", err)
		}
		vm.Reset()
	}

	// Extract learned weights and bias
	var model LinearModel
	wVal := w.Value().Data().([]float64)
	for j := 0; j < numFeatures; j++ {
		model.Weights[j] = wVal[j]
	}
	model.Bias = wVal[numFeatures]

	return model
}

func (p *WeatherPredictor) predictValue(model LinearModel, features [numFeatures]float64, tMean, tStd float64) float64 {
	pred := model.Bias
	for j := 0; j < numFeatures; j++ {
		nf := (features[j] - p.FeatureMean[j]) / p.FeatureStd[j]
		pred += model.Weights[j] * nf
	}
	return pred*tStd + tMean
}

func (p *WeatherPredictor) predictWeather(month int, tempAvg float64) (string, string) {
	bucket := int(tempAvg) / 5

	// Try exact month + temp bucket first
	if buckets, ok := p.WeatherStats[month]; ok {
		if counts, ok := buckets[bucket]; ok {
			return dominantWeather(counts)
		}
		// Fall back to all temps for this month
		merged := make(map[string]int)
		for _, counts := range buckets {
			for w, c := range counts {
				merged[w] += c
			}
		}
		return dominantWeather(merged)
	}

	return "Clear", "clear sky"
}

func dominantWeather(counts map[string]int) (string, string) {
	best := ""
	bestCount := 0
	for w, c := range counts {
		if c > bestCount {
			bestCount = c
			best = w
		}
	}
	descMap := map[string]string{
		"Clear":        "clear sky",
		"Clouds":       "overcast clouds",
		"Rain":         "moderate rain",
		"Drizzle":      "light drizzle",
		"Snow":         "light snow",
		"Thunderstorm": "thunderstorm",
		"Mist":         "mist",
		"Fog":          "fog",
		"Haze":         "haze",
	}
	desc := descMap[best]
	if desc == "" {
		desc = strings.ToLower(best)
	}
	return best, desc
}

func (p *WeatherPredictor) todayProxy() DailyRecord {
	today := time.Now()
	todayDOY := today.YearDay()

	var temp, dewPoint, feelsLike, pressure, humidity, windSpeed, clouds float64
	var tempMin, tempMax float64
	count := 0

	for _, d := range p.DailyData {
		if d.Date.YearDay() == todayDOY {
			temp += d.Temp
			dewPoint += d.DewPoint
			feelsLike += d.FeelsLike
			pressure += d.Pressure
			humidity += d.Humidity
			windSpeed += d.WindSpeed
			clouds += d.Clouds
			tempMin += d.TempMin
			tempMax += d.TempMax
			count++
		}
	}

	if count == 0 {
		return p.DailyData[len(p.DailyData)-1]
	}

	n := float64(count)
	return DailyRecord{
		Date:      today,
		Temp:      temp / n,
		TempMin:   tempMin / n,
		TempMax:   tempMax / n,
		FeelsLike: feelsLike / n,
		DewPoint:  dewPoint / n,
		Pressure:  pressure / n,
		Humidity:  humidity / n,
		WindSpeed: windSpeed / n,
		Clouds:    clouds / n,
	}
}

func (p *WeatherPredictor) PredictTomorrow() Prediction {
	today := p.todayProxy()
	tomorrow := time.Now().AddDate(0, 0, 1)
	features := extractFeatures(today)

	tempAvg := p.predictValue(p.TempModel, features, p.TempMean, p.TempStd)
	tempMin := p.predictValue(p.TempMinModel, features, p.TempMinMean, p.TempMinStd)
	tempMax := p.predictValue(p.TempMaxModel, features, p.TempMaxMean, p.TempMaxStd)
	feelsLike := p.predictValue(p.FeelsLikeModel, features, p.FeelsLikeMean, p.FeelsLikeStd)
	humidity := p.predictValue(p.HumidityModel, features, p.HumidityMean, p.HumidityStd)
	pressure := p.predictValue(p.PressureModel, features, p.PressureMean, p.PressureStd)
	windSpeed := p.predictValue(p.WindModel, features, p.WindMean, p.WindStd)

	weatherMain, weatherDesc := p.predictWeather(int(tomorrow.Month()), tempAvg)

	round := func(v float64) float64 { return math.Round(v*10) / 10 }

	return Prediction{
		Date:        tomorrow.Format("2006-01-02"),
		City:        "Kavadarci",
		TempAvg:     round(tempAvg),
		TempMin:     round(tempMin),
		TempMax:     round(tempMax),
		FeelsLike:   round(feelsLike),
		Humidity:    round(humidity),
		WindSpeed:   round(windSpeed),
		Pressure:    round(pressure),
		WeatherMain: weatherMain,
		WeatherDesc: weatherDesc,
	}
}

func main() {
	filePath := "latest_dataset_open_weather_kav_1979-2024.csv"

	fmt.Println("Loading and aggregating weather data...")
	days, err := loadAndAggregate(filePath)
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}
	fmt.Printf("Loaded %d daily records (%s to %s)\n", len(days),
		days[0].Date.Format("2006-01-02"),
		days[len(days)-1].Date.Format("2006-01-02"))

	// Build training pairs: features from day[i] -> targets from day[i+1]
	n := len(days) - 1
	features := make([][numFeatures]float64, n)
	tempTargets := make([]float64, n)
	tempMinTargets := make([]float64, n)
	tempMaxTargets := make([]float64, n)
	feelsLikeTargets := make([]float64, n)
	humidityTargets := make([]float64, n)
	pressureTargets := make([]float64, n)
	windTargets := make([]float64, n)

	for i := 0; i < n; i++ {
		features[i] = extractFeatures(days[i])
		tempTargets[i] = days[i+1].Temp
		tempMinTargets[i] = days[i+1].TempMin
		tempMaxTargets[i] = days[i+1].TempMax
		feelsLikeTargets[i] = days[i+1].FeelsLike
		humidityTargets[i] = days[i+1].Humidity
		pressureTargets[i] = days[i+1].Pressure
		windTargets[i] = days[i+1].WindSpeed
	}

	var fMean, fStd [numFeatures]float64
	for j := 0; j < numFeatures; j++ {
		col := make([]float64, n)
		for i := 0; i < n; i++ {
			col[i] = features[i][j]
		}
		fMean[j], fStd[j] = meanStd(col)
	}

	tempMean, tempStd := meanStd(tempTargets)
	tempMinMean, tempMinStd := meanStd(tempMinTargets)
	tempMaxMean, tempMaxStd := meanStd(tempMaxTargets)
	feelsLikeMean, feelsLikeStd := meanStd(feelsLikeTargets)
	humidityMean, humidityStd := meanStd(humidityTargets)
	pressureMean, pressureStd := meanStd(pressureTargets)
	windMean, windStd := meanStd(windTargets)

	// Build weather condition stats: month -> temp_bucket -> weather -> count
	weatherStats := make(map[int]map[int]map[string]int)
	for _, d := range days {
		m := int(d.Date.Month())
		bucket := int(d.Temp) / 5
		if weatherStats[m] == nil {
			weatherStats[m] = make(map[int]map[string]int)
		}
		if weatherStats[m][bucket] == nil {
			weatherStats[m][bucket] = make(map[string]int)
		}
		weatherStats[m][bucket][d.WeatherMain]++
	}

	epochs := 200
	lr := 0.01

	fmt.Println("Training models...")
	tempModel := trainLinearRegression(features, tempTargets, fMean, fStd, tempMean, tempStd, epochs, lr)
	tempMinModel := trainLinearRegression(features, tempMinTargets, fMean, fStd, tempMinMean, tempMinStd, epochs, lr)
	tempMaxModel := trainLinearRegression(features, tempMaxTargets, fMean, fStd, tempMaxMean, tempMaxStd, epochs, lr)
	feelsLikeModel := trainLinearRegression(features, feelsLikeTargets, fMean, fStd, feelsLikeMean, feelsLikeStd, epochs, lr)
	humidityModel := trainLinearRegression(features, humidityTargets, fMean, fStd, humidityMean, humidityStd, epochs, lr)
	pressureModel := trainLinearRegression(features, pressureTargets, fMean, fStd, pressureMean, pressureStd, epochs, lr)
	windModel := trainLinearRegression(features, windTargets, fMean, fStd, windMean, windStd, epochs, lr)

	predictor := &WeatherPredictor{
		TempModel:      tempModel,
		TempMinModel:   tempMinModel,
		TempMaxModel:   tempMaxModel,
		FeelsLikeModel: feelsLikeModel,
		HumidityModel:  humidityModel,
		PressureModel:  pressureModel,
		WindModel:      windModel,
		FeatureMean:    fMean,
		FeatureStd:     fStd,
		TempMean:       tempMean,
		TempStd:        tempStd,
		TempMinMean:    tempMinMean,
		TempMinStd:     tempMinStd,
		TempMaxMean:    tempMaxMean,
		TempMaxStd:     tempMaxStd,
		FeelsLikeMean:  feelsLikeMean,
		FeelsLikeStd:   feelsLikeStd,
		HumidityMean:   humidityMean,
		HumidityStd:    humidityStd,
		PressureMean:   pressureMean,
		PressureStd:    pressureStd,
		WindMean:       windMean,
		WindStd:        windStd,
		DailyData:      days,
		WeatherStats:   weatherStats,
	}

	pred := predictor.PredictTomorrow()
	fmt.Printf("\nPrediction for %s, %s:\n", pred.City, pred.Date)
	fmt.Printf("  Temperature: %.1f°C (feels like %.1f°C)\n", pred.TempAvg, pred.FeelsLike)
	fmt.Printf("  Min: %.1f°C, Max: %.1f°C\n", pred.TempMin, pred.TempMax)
	fmt.Printf("  Humidity: %.0f%%, Wind: %.1f m/s, Pressure: %.0f hPa\n", pred.Humidity, pred.WindSpeed, pred.Pressure)
	fmt.Printf("  Conditions: %s (%s)\n", pred.WeatherMain, pred.WeatherDesc)

	fmt.Println("\nServer running on http://localhost:8080")
	fmt.Println("  curl localhost:8080                  # text output")
	fmt.Println("  curl localhost:8080/predict           # JSON output")
	fmt.Println("  curl localhost:8080/predict/tomorrow   # JSON output")

	serveJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		enc.Encode(predictor.PredictTomorrow())
	}

	http.HandleFunc("/predict/tomorrow", serveJSON)
	http.HandleFunc("/predict", serveJSON)

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		p := predictor.PredictTomorrow()
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		fmt.Fprintf(w, "Weather prediction for %s, %s\n", p.City, p.Date)
		fmt.Fprintf(w, "========================================\n")
		fmt.Fprintf(w, "Temperature:  %.1f°C (feels like %.1f°C)\n", p.TempAvg, p.FeelsLike)
		fmt.Fprintf(w, "Min / Max:    %.1f°C / %.1f°C\n", p.TempMin, p.TempMax)
		fmt.Fprintf(w, "Humidity:     %.0f%%\n", p.Humidity)
		fmt.Fprintf(w, "Wind speed:   %.1f m/s\n", p.WindSpeed)
		fmt.Fprintf(w, "Pressure:     %.0f hPa\n", p.Pressure)
		fmt.Fprintf(w, "Conditions:   %s (%s)\n", p.WeatherMain, p.WeatherDesc)
	})

	log.Fatal(http.ListenAndServe(":8080", nil))
}
