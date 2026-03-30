##############################################
### To test link flaps and speed downgrade ###
##############################################

# Advertise values
# 10 Mbps
# Mode	Value
# 10baseT Half	0x001
# 10baseT Full	0x002

# 100 Mbps
# Mode	Value
# 100baseT Half	0x004
# 100baseT Full	0x008

# 1000 Mbps (Gigabit)
# Mode	Value
# 1000baseT Half	0x010
# 1000baseT Full	0x020

# change speed on interface will cause link flap (DOWN & UP)
ethtool -s <interface> advertise 0x002 # change speed to 10BaseT full

# To cancel commands run
ethtool -s <interface> autoneg on

# verify with
ethtool <interface>