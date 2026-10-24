#RUN: vmd -dispdev text -gro input.gro md_noPBC.xtc -e extract-frame.tcl
#RUN;vmd -dispdev text input.gro centered.xtc -e extract-frame.tcl, I used thsi for pbc extraction
# 1. Load pbctools package
package require pbctools

# 2. Apply PBC wrapping across ALL trajectory frames
# -compound fragment: Keeps molecular units/monomers intact without breaking bonds across box edges
# -center origin: Re-centers the coordinate system
# -all: Processes all 20,000 frames in memory
puts "Applying PBC wrapping to keep dimer intact across all frames..."
pbc wrap -compound fragment -center origin -all

# 3. Ensure target directory exists and change to it
file mkdir frame_extract
cd frame_extract

# 4. Get total number of frames from the loaded molecule
set num_frames [molinfo top get numframes]
puts "Extracting $num_frames PBC-wrapped frames..."

# 5. Loop through all frames and write individual XYZ files
for {set i 0} {$i < $num_frames} {incr i} {
    set sel [atomselect top "all" frame $i]
    set filename [format "geom_%d.xyz" $i]

    animate write xyz $filename beg $i end $i sel $sel
    $sel delete

    if {$i % 1000 == 0} {
        puts "Saved $filename ($i / [expr {$num_frames - 1}])"
    }
}

puts "PBC-wrapped extraction complete!"
exit
