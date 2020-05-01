// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"strconv"
)

func getGPUCount(bin string) int {
	query := fmt.Sprintf("--query-gpu=%s", "count")
	gpu := fmt.Sprintf("--id=%d", 0)
	opts := []string{"--format=noheader,nounits,csv", query, gpu}

	ret, err := exec.Command(bin, opts...).CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %s", err, ret)
		return 0
	}
	retS := strings.TrimSuffix(string(ret), "\n")
	retI, errI := strconv.Atoi(retS)
	if errI != nil {
		fmt.Fprintf(os.Stderr, "%s: %s", errI, retI)
		return 0
	}
	return retI
}

func getResult(bin string, metric string, verbose bool, gpuId int) string {
	query := fmt.Sprintf("--query-gpu=%s", metric)
	gpu := fmt.Sprintf("--id=%d", gpuId)
	opts := []string{"--format=noheader,nounits,csv", query, gpu}

	if verbose {
		fmt.Print("Going to run ")
		fmt.Print(bin)
		fmt.Println(" with ")
		fmt.Println(opts)
	}
	ret, err := exec.Command(bin, opts...).CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %s", err, ret)
		return ""
	}
	return string(ret)
}

func main() {
	binPath := flag.String("bin", "/usr/bin/nvidia-smi", "nvidia-smi full path")
	verbose := flag.Bool("verbose", false, "display some things")

	flag.Parse()

	if _, err := os.Stat(*binPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Bin path does not exists: %s\n", *binPath)
		return // exit
	}

	metrics := "fan.speed,memory.total,memory.used,memory.free,pstate,temperature.gpu,name,uuid,compute_mode,utilization.gpu,utilization.memory,index"
	gpuCount := getGPUCount(*binPath)
	// No need to collect data if nvidia-smi reports no
	if gpuCount == 0 {
		return //exit
	}

	for i := 0; i < gpuCount; i++ {
		results := getResult(*binPath, metrics, *verbose, i)

		if results == "" {
			return // exit
		}

		splitResults := strings.Split(results, ",")

		fmt.Printf("nvidiasmi,uuid=%s ", strings.TrimSpace(splitResults[7])) // it should be available ... if no, you have some problems

		fmt.Printf("gpu_name=\"%s\",", strings.TrimSpace(splitResults[6]))
		fmt.Printf("gpu_compute_mode=\"%s\",", strings.TrimSpace(splitResults[8]))

		fmt.Printf("memory_total=%s,", strings.TrimSpace(splitResults[1])) // they
		fmt.Printf("memory_used=%s,", strings.TrimSpace(splitResults[2]))  // are
		fmt.Printf("memory_free=%s,", strings.TrimSpace(splitResults[3]))  // MiB

		fmt.Printf("pstate=%s,", strings.TrimSpace(strings.Replace(splitResults[4], "P", "", -1))) // strip the P
		fmt.Printf("temperature=%s,", strings.TrimSpace(splitResults[5])) // in degrees Celcius

		fmt.Printf("utilization_gpu=%s,", strings.TrimSpace(splitResults[9])) // it's a % 0-100
		fmt.Printf("utilization_memory=%s,", strings.TrimSpace(splitResults[10])) // it's a % 0-100

		fmt.Printf("gpu_index=%s\n", strings.TrimSpace(splitResults[11])) // index as reported by smi
	}
}
