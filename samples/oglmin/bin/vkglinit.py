import subprocess
from array import array
import math

output = open('output.txt', 'w')

vulkan_args = ['vkglinit', 'vulkan', 'wait']
opengl_args = ['vkglinit', 'opengl', 'wait']

vulkan_times = array('f')
opengl_times = array('f')

# Run once and ignore result; if this is run right after driver install
# then this first time will take longer than subsequent runs just because
# the driver DLL isn't cached
p0 = subprocess.Popen(vulkan_args, stdout=subprocess.PIPE)
p0.stdout.readline()
p0.terminate()

# Test vulkan
p0 = subprocess.Popen(vulkan_args, stdout=subprocess.PIPE)
vulkan_times.append(float(p0.stdout.readline()))
p1 = subprocess.Popen(vulkan_args, stdout=subprocess.PIPE)
vulkan_times.append(float(p1.stdout.readline()))
p2 = subprocess.Popen(vulkan_args, stdout=subprocess.PIPE)
vulkan_times.append(float(p2.stdout.readline()))
p3 = subprocess.Popen(vulkan_args, stdout=subprocess.PIPE)
vulkan_times.append(float(p3.stdout.readline()))

p0.terminate()
p1.terminate()
p2.terminate()
p3.terminate()

# Test opengl
p0 = subprocess.Popen(opengl_args, stdout=subprocess.PIPE)
opengl_times.append(float(p0.stdout.readline()))
p1 = subprocess.Popen(opengl_args, stdout=subprocess.PIPE)
opengl_times.append(float(p1.stdout.readline()))
p2 = subprocess.Popen(opengl_args, stdout=subprocess.PIPE)
opengl_times.append(float(p2.stdout.readline()))
p3 = subprocess.Popen(opengl_args, stdout=subprocess.PIPE)
opengl_times.append(float(p3.stdout.readline()))

p0.terminate()
p1.terminate()
p2.terminate()
p3.terminate()

vulkan_avg = sum(vulkan_times) / len(vulkan_times)
vulkan_sd = 0.0
for t in vulkan_times:
    delta = t - vulkan_avg
    vulkan_sd += delta * delta
vulkan_sd = math.sqrt(vulkan_sd)

opengl_avg = sum(opengl_times) / len(opengl_times)
opengl_sd = 0.0
for t in opengl_times:
    delta = t - opengl_avg
    opengl_sd += delta * delta
opengl_sd = math.sqrt(opengl_sd)

output.write('Vulkan (Mean, SD): {:.2f} ms, {:.2f} ms\n'.format(vulkan_avg, vulkan_sd))
output.write('OpenGL (Mean, SD): {:.2f} ms, {:.2f} ms\n'.format(opengl_avg, opengl_sd))

output.close()
