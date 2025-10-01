package de.jlkp.experimental;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class TestThreads extends Thread {
    @Override
    public void run() {
        for (int i = 0; i < 5; i++) {
            System.out.println("Thread " + this.getName() + " - Count: " + i);
            try {
                Thread.sleep(500); // Sleep for 1 second
            } catch (InterruptedException e) {
                log.info("Thread interrupted: {}", e.getMessage());
            }
        }
    }

}
