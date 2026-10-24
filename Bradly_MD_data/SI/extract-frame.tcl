# extract_frames.tcl
#RUN command:vmd -dispdev text input.gro centeredP1.xtc -e extract-frame.tcl
# 1. Ensure target directory exists and enter it
file mkdir frameP1_extract
cd frameP1_extract

# 2. Get total number of frames from loaded molecule
set num_frames [molinfo top get numframes]
puts "Total frames loaded: $num_frames"

# Option A: Fast batch export using 'animate write' directly (Recommended)
# Writes all frames to a single multi-frame trajectory file (20,000 frames)
# Useful for high-throughput dataset preparation without file system bloat
# animate write xyz all_dimer_geometries.xyz beg 0 end [expr {$num_frames - 1}] waitfor all top


# Option B: Extract into 20,000 separate single-frame XYZ files
puts "Extracting individual XYZ files..."
set sel [atomselect top "all"]

for {set i 0} {$i < $num_frames} {incr i} {
    $sel frame $i
    set filename [format "geom_%05d.xyz" $i]
    $sel writexyz $filename

    if {$i % 1000 == 0} {
        puts "Saved $filename ($i / [expr {$num_frames - 1}])"
    }
}

$sel delete
puts "Extraction complete!"
exit
