SELECT kindOfShot, COUNT(*) FROM data WHERE kindOfShot != "NONE" AND kindOfShot != "" AND locationIsInternal != "True" AND eventName = "CaptureStarted" GROUP BY kindOfShot;
